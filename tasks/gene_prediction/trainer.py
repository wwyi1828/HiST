import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig

from lib.utils.coord_aug import update_coord_aug_multiplier
from lib.utils.metrics import compute_gene_metrics
from lib.utils.wandb_helper import create_logger


def train_one_epoch(model, data_loader, optimizer, device, cfg, epoch=None):
    model.train()
    total_loss = 0
    desc = f"Epoch {epoch}/{cfg.training.epochs}" if epoch is not None else "Training"
    progress_bar = tqdm(data_loader, desc=desc, leave=False)

    for batch in progress_bar:
        img_feats = batch.mor_feats.to(device)
        gene_targets = batch.mol_feats.to(device)
        final_cords = batch.final_cords.to(device)
        total_samples = img_feats.size(0)
        sub_batch_size = (cfg.training.sub_batch_size
                          if cfg.training.sub_batch_size is not None else total_samples)
        accumulate_steps = cfg.training.accumulate_steps

        optimizer.zero_grad()
        batch_loss = 0
        num_sub_batches = 0
        extended_size = int(sub_batch_size * 1.1)
        current_idx = 0

        while current_idx < total_samples:
            remaining = total_samples - current_idx
            if remaining >= extended_size:
                start_offset = random.randint(0, extended_size - sub_batch_size)
                selected_indices = list(range(
                    current_idx + start_offset,
                    current_idx + start_offset + sub_batch_size,
                ))
            elif remaining >= sub_batch_size:
                selected_indices = list(range(current_idx, current_idx + sub_batch_size))
            else:
                backward_needed = sub_batch_size - remaining
                if current_idx >= backward_needed:
                    selected_indices = (list(range(current_idx - backward_needed, current_idx))
                                        + list(range(current_idx, total_samples)))
                else:
                    selected_indices = list(range(total_samples))

            log_prediction = model(final_cords[selected_indices], img_feats[selected_indices])
            loss = F.mse_loss(log_prediction, torch.log1p(gene_targets[selected_indices]))
            (loss / accumulate_steps).backward()
            batch_loss += loss.item()
            num_sub_batches += 1

            current_idx = total_samples if remaining <= extended_size else current_idx + extended_size
            if num_sub_batches % accumulate_steps == 0 or current_idx >= total_samples:
                optimizer.step()
                optimizer.zero_grad()

        avg_batch_loss = batch_loss / num_sub_batches
        total_loss += avg_batch_loss
        progress_bar.set_postfix(loss=avg_batch_loss)

    return total_loss / len(data_loader)


def evaluate(
    model,
    data_loader,
    device,
    cfg,
    compute_metrics=True,
    save_dir=None,
    slide_token_save_dir=None,
):
    model.eval()
    total_loss = 0
    eval_metrics = cfg.training.eval_metrics
    slide_metrics = {metric: [] for metric in eval_metrics}
    all_predictions = []
    all_targets = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if slide_token_save_dir:
        os.makedirs(slide_token_save_dir, exist_ok=True)
        token_init_types = np.asarray(
            [str(t) for t in model.slide_encoder.token_init_types], dtype='S',
        )

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            img_feats = batch.mor_feats.to(device)
            gene_targets = batch.mol_feats.to(device)
            final_cords = batch.final_cords.to(device)
            slide_token = None
            if slide_token_save_dir:
                log_prediction, slide_token = model(final_cords, img_feats, return_slide_token=True)
            else:
                log_prediction = model(final_cords, img_feats)
            loss = F.mse_loss(log_prediction, torch.log1p(gene_targets))
            total_loss += loss.item()

            if not compute_metrics:
                continue

            predictions = torch.expm1(log_prediction).relu().cpu().numpy()
            targets = gene_targets.cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets)

            if save_dir:
                with h5py.File(os.path.join(save_dir, f"{batch.slide_name}.h5"), 'w') as file:
                    file.create_dataset('predictions', data=predictions)
                    file.create_dataset('targets', data=targets)
                    file.create_dataset('coords', data=batch.final_cords.numpy())
                    file.create_dataset('gene_names', data=np.asarray(batch.gene_names, dtype='S'))
            if slide_token_save_dir and slide_token is not None:
                with h5py.File(os.path.join(slide_token_save_dir, f"{batch.slide_name}.h5"), 'w') as file:
                    file.create_dataset('slide_token', data=slide_token.cpu().numpy())
                    file.create_dataset('token_init_types', data=token_init_types)

            values = compute_gene_metrics(targets.ravel(), predictions.ravel(), metrics=eval_metrics)
            for name, value in values.items():
                slide_metrics[name].append(value)

    mean_metrics = {}
    std_metrics = {}
    spot_metrics = {}
    if compute_metrics:
        mean_metrics = {name: np.mean(values) for name, values in slide_metrics.items()}
        std_metrics = {name: np.std(values) for name, values in slide_metrics.items()}
        spot_metrics = compute_gene_metrics(
            np.vstack(all_targets).ravel(), np.vstack(all_predictions).ravel(), metrics=eval_metrics,
        )
        print("\nSlide-Level Evaluation Metrics (Mean ± Std):")
        for name in mean_metrics:
            print(f"{name}: {mean_metrics[name]:.4f} ± {std_metrics[name]:.4f}")
        print("\nSpot-Level Evaluation Metrics:")
        for name, value in spot_metrics.items():
            print(f"{name}: {value:.4f}")

    return total_loss / len(data_loader), mean_metrics, std_metrics, spot_metrics


def train_gene_prediction(
    cfg: DictConfig,
    train_loader,
    test_loader,
    model,
    model_config,
):

    print(f"Training prediction model with backbone: {cfg.patch_backbone}")
    print(f"Train mode: {cfg.model.train_mode}")

    if cfg.model.train_mode == 'lora':
        print(f"LoRA config: rank={cfg.model.lora_rank}, alpha={cfg.model.lora_alpha}")
        print(f"Learning rate: base={cfg.training.lr:.2e}, LoRA={cfg.training.lr * cfg.model.lora_lr_mult:.2e}")
    else:
        print(f"Learning rate: {cfg.training.lr:.2e}, Epochs: {cfg.training.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger = create_logger(cfg)

    model.to(device)

    if cfg.training.slide_pretrained:
        ckpt_root = os.environ.get('HIST_CKPT_ROOT', './checkpoints')
        checkpoint_path = os.path.join(
            ckpt_root,
            f'frozen_211_long_{cfg.patch_backbone}_clip_{cfg.dataset}_{cfg.test_fold}.pt',
        )
        slide_encoder_path = os.path.join(checkpoint_path, 'slide_encoder.pt')
        if os.path.exists(slide_encoder_path):
            print(f"Loading slide encoder from: {slide_encoder_path}")
            slide_encoder_state_dict = torch.load(slide_encoder_path, map_location=device)
            model.slide_encoder.load_state_dict(slide_encoder_state_dict, strict=False)
            print(f"Slide encoder loaded successfully")
        else:
            print(f"Warning: Slide encoder checkpoint not found at {slide_encoder_path}")

    if cfg.training.get('init_bias_with_mean', False):
        gene_means = []
        with torch.no_grad():
            for batch in train_loader:
                gene_means.append(torch.log1p(batch.mol_feats.to(device)))
            gene_mean = torch.cat(gene_means, dim=0).mean(dim=0)

        final_linear = model.decoder.blocks[-1].modules_dict['convs_0'][0].conv_block.linear[-1]
        final_linear.bias.data.copy_(gene_mean)

    if cfg.model.train_mode == 'lora':
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        print(f"Optimizer: LoRA params={len(lora_params)} (lr={cfg.training.lr * cfg.model.lora_lr_mult:.2e}), Other params={len(other_params)} (lr={cfg.training.lr:.2e})")

        param_groups = [
            {'params': lora_params, 'lr': cfg.training.lr * cfg.model.lora_lr_mult},
            {'params': other_params, 'lr': cfg.training.lr}
        ]
        optimizer = AdamW(param_groups, weight_decay=cfg.training.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    eta_min_factor = cfg.training.get('eta_min_factor', 0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * eta_min_factor)

    for epoch in range(1, cfg.training.epochs + 1):
        update_coord_aug_multiplier(cfg, train_loader, epoch, cfg.training.epochs)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)

        compute_metrics = (epoch == cfg.training.epochs)

        if cfg.training.eval_metric_freq is None:
            should_evaluate = (epoch == cfg.training.epochs)
        else:
            should_evaluate = (epoch % cfg.training.eval_metric_freq == 0) or (epoch == cfg.training.epochs)

        if should_evaluate:
            run_name = cfg.dataset_group if cfg.dataset_group is not None else cfg.dataset
            save_dir = (
                os.path.join(
                    cfg.logging.results.dir,
                    "predictions",
                    cfg.logging.results.save_path,
                    model_config,
                    f"{run_name}_fold{cfg.test_fold}",
                )
                if compute_metrics and cfg.logging.results.get("save_predictions", True) else None
            )
            slide_token_save_dir = (
                os.path.join(
                    cfg.logging.results.dir,
                    "slide_token",
                    cfg.logging.results.save_path,
                    model_config,
                    f"{run_name}_fold{cfg.test_fold}",
                )
                if compute_metrics and cfg.logging.results.get("save_predictions", True) else None
            )
            val_loss, avg_slide_metrics, _, spot_level_metrics = evaluate(
                model,
                test_loader,
                device,
                cfg,
                compute_metrics=compute_metrics,
                save_dir=save_dir,
                slide_token_save_dir=slide_token_save_dir,
            )

            if compute_metrics:
                final_metrics = {
                    'spot_metrics': spot_level_metrics,
                    'slide_avg_metrics': {k: float(v) for k, v in avg_slide_metrics.items()},
                }

            if logger:
                logger.log_epoch(epoch, {"train/loss": float(train_loss), "val/loss": float(val_loss)})
        else:
            if logger:
                logger.log_epoch(epoch, {"train/loss": float(train_loss)})

        scheduler.step()

    print("Training completed!")

    if logger:
        logger.log_final(final_metrics)
        logger.finish()

    return final_metrics
