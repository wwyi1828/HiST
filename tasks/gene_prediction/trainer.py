import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig

from lib.utils.coord_aug import update_coord_aug_multiplier
import h5py
from lib.utils.metrics import compute_gene_metrics
from lib.utils.wandb_helper import create_logger
from .model import PredictionModel

def train_one_epoch(model, data_loader, optimizer, device, cfg, gene_transform, epoch=None):

    model.train()
    total_loss = 0

    desc = f"Epoch {epoch}/{cfg.training.epochs}" if epoch is not None else "Training"
    progress_bar = tqdm(data_loader, desc=desc, leave=False)

    for batch_idx, batch in enumerate(progress_bar):

        if isinstance(batch.mor_feats, torch.Tensor):
            img_feats = batch.mor_feats.to(device)
        if isinstance(batch.mol_feats, torch.Tensor):
            gene_hvg = batch.mol_feats.to(device)
            gene_hvg = gene_transform(gene_hvg)

        if isinstance(batch.final_cords, torch.Tensor):
            final_cords = batch.final_cords.to(device)

        total_samples = img_feats.size(0)
        sub_batch_size = cfg.training.sub_batch_size if cfg.training.sub_batch_size is not None else total_samples
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
                    current_idx + start_offset + sub_batch_size
                ))
            elif remaining >= sub_batch_size:
                selected_indices = list(range(current_idx, current_idx + sub_batch_size))
            else:
                backward_needed = sub_batch_size - remaining

                if current_idx >= backward_needed:
                    backward_indices = list(range(current_idx - backward_needed, current_idx))
                    forward_indices = list(range(current_idx, total_samples))
                    selected_indices = backward_indices + forward_indices
                else:
                    selected_indices = list(range(total_samples))

            sub_batch = type('', (), {})()
            sub_batch.mor_feats = img_feats[selected_indices]
            sub_batch.mol_feats = gene_hvg[selected_indices]
            sub_batch.cords = final_cords[selected_indices]

            _, loss = model(sub_batch)
            if loss is not None:
                scaled_loss = loss / accumulate_steps
                scaled_loss.backward()
                batch_loss += loss.item()
                num_sub_batches += 1

            if remaining <= extended_size:
                current_idx = total_samples
            else:
                current_idx += extended_size

            is_last_sub_batch = (current_idx >= total_samples)
            should_step = (num_sub_batches % accumulate_steps == 0) or is_last_sub_batch

            if should_step and num_sub_batches > 0:
                optimizer.step()
                optimizer.zero_grad()

        if num_sub_batches > 0:
            avg_batch_loss = batch_loss / num_sub_batches
            total_loss += avg_batch_loss
            progress_bar.set_postfix(loss=avg_batch_loss)

    return total_loss / len(data_loader)

def evaluate(
    model,
    data_loader,
    device,
    cfg,
    gene_transform,
    compute_metrics=True,
    save_dir=None,
    slide_token_save_dir=None,
):

    model.eval()
    total_loss = 0
    num_batches = 0

    eval_metrics = cfg.training.eval_metrics
    slide_metrics = {metric: [] for metric in eval_metrics}

    all_predictions = []
    all_targets = []

    gene_names = None
    token_init_types = None
    if save_dir or slide_token_save_dir:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if slide_token_save_dir:
            os.makedirs(slide_token_save_dir, exist_ok=True)
        sample = data_loader.dataset[0]
        gene_names_full = sample.union_gene_names
        item = data_loader.dataset.slide_items[0]
        gene_type = item[2]
        num_genes_val = item[3]
        if gene_type == 'hvg' and hasattr(sample, 'hvg_indices'):
            indices = sample.hvg_indices[:num_genes_val].tolist() if num_genes_val else sample.hvg_indices.tolist()
            gene_names = [gene_names_full[i] for i in indices]
        elif gene_type == 'heg' and hasattr(sample, 'heg_indices'):
            indices = sample.heg_indices[:num_genes_val].tolist() if num_genes_val else sample.heg_indices.tolist()
            gene_names = [gene_names_full[i] for i in indices]
        else:
            gene_names = gene_names_full
        token_init_types = np.asarray(
            [str(t) for t in getattr(model.slide_encoder, 'token_init_types', [])],
            dtype='S',
        )

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):

            if isinstance(batch.mor_feats, torch.Tensor):
                img_feats = batch.mor_feats.to(device)
            if isinstance(batch.mol_feats, torch.Tensor):
                gene_hvg = batch.mol_feats.to(device)
                gene_hvg = gene_transform(gene_hvg)

            if isinstance(batch.final_cords, torch.Tensor):
                final_cords = batch.final_cords.to(device)

            total_samples = img_feats.size(0)
            sub_batch_size = total_samples

            batch_loss = 0
            num_sub_batches = 0

            slide_predictions = []
            slide_targets = []
            slide_token = None

            for start_idx in range(0, total_samples, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, total_samples)

                sub_batch = type('', (), {})()
                sub_batch.mor_feats = img_feats[start_idx:end_idx]
                sub_batch.mol_feats = gene_hvg[start_idx:end_idx]
                sub_batch.cords = final_cords[start_idx:end_idx]

                if slide_token_save_dir:
                    predictions, loss, slide_token = model(sub_batch, return_slide_token=True)
                else:
                    predictions, loss = model(sub_batch)

                if compute_metrics:
                    slide_predictions.append(predictions.cpu().numpy())
                    slide_targets.append(sub_batch.mol_feats.cpu().numpy())

                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(sub_batch.mol_feats.cpu().numpy())

                if loss is not None:
                    batch_loss += loss.item()
                    num_sub_batches += 1

            if num_sub_batches > 0:
                avg_batch_loss = batch_loss / num_sub_batches
                total_loss += avg_batch_loss
                num_batches += 1

                if compute_metrics:
                    slide_predictions = np.vstack(slide_predictions) if slide_predictions else np.array([])
                    slide_targets = np.vstack(slide_targets) if slide_targets else np.array([])

                    if save_dir:
                        with h5py.File(os.path.join(save_dir, f"{batch.slide_name}.h5"), 'w') as f:
                            f.create_dataset('predictions', data=slide_predictions)
                            f.create_dataset('targets', data=slide_targets)
                            f.create_dataset('coords', data=batch.final_cords.cpu().numpy())
                            f.create_dataset('gene_names', data=np.array(gene_names, dtype='S'))
                    if slide_token_save_dir and slide_token is not None:
                        with h5py.File(os.path.join(slide_token_save_dir, f"{batch.slide_name}.h5"), 'w') as f:
                            f.create_dataset('slide_token', data=slide_token.cpu().numpy())
                            f.create_dataset('token_init_types', data=token_init_types)

                    slide_metric_values = compute_gene_metrics(
                        slide_targets.flatten(),
                        slide_predictions.flatten(),
                        binary_threshold=0,
                        log_space=False,
                        metrics=eval_metrics
                    )

                    for metric_name, metric_value in slide_metric_values.items():
                        slide_metrics[metric_name].append(metric_value)

    avg_slide_metrics = {metric_name: np.mean(values) for metric_name, values in slide_metrics.items()} if compute_metrics else {}
    std_slide_metrics = {metric_name: np.std(values) for metric_name, values in slide_metrics.items()} if compute_metrics else {}

    if compute_metrics:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        spot_level_metrics = compute_gene_metrics(
            all_targets.flatten(),
            all_predictions.flatten(),
            binary_threshold=0,
            log_space=False,
            metrics=eval_metrics
        )

        print("\nSlide-Level Evaluation Metrics (Mean ± Std):")
        for metric_name in avg_slide_metrics:
            print(f"{metric_name}: {avg_slide_metrics[metric_name]:.4f} ± {std_slide_metrics[metric_name]:.4f}")

        print("\nSpot-Level Evaluation Metrics:")
        for metric_name, metric_value in spot_level_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    else:
        spot_level_metrics = {}

    return total_loss / max(num_batches, 1), avg_slide_metrics, std_slide_metrics, spot_level_metrics

def train_gene_prediction(
    cfg: DictConfig,
    train_loader,
    test_loader,
    image_suffix,
    gene_transform,
    padder,
    slide_encoder,
    decoder,
    segmentor,
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

    model = PredictionModel(
        padder=padder,
        slide_encoder=slide_encoder,
        decoder=decoder,
        segmentor=segmentor,
        image_suffix=image_suffix,
        cfg=cfg
    ).to(device)

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

    best_val_loss = float('inf')
    final_spot_metrics = None
    final_avg_slide_metrics = None
    final_std_slide_metrics = None

    for epoch in range(1, cfg.training.epochs + 1):
        update_coord_aug_multiplier(cfg, train_loader, epoch, cfg.training.epochs)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, gene_transform, epoch)

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
            val_loss, avg_slide_metrics, std_slide_metrics, spot_level_metrics = evaluate(
                model,
                test_loader,
                device,
                cfg,
                gene_transform,
                compute_metrics=compute_metrics,
                save_dir=save_dir,
                slide_token_save_dir=slide_token_save_dir,
            )

            if compute_metrics:
                final_spot_metrics = spot_level_metrics
                final_avg_slide_metrics = avg_slide_metrics
                final_std_slide_metrics = std_slide_metrics

            if logger:
                logger.log_epoch(epoch, {"train/loss": float(train_loss), "val/loss": float(val_loss)})
        else:
            if logger:
                logger.log_epoch(epoch, {"train/loss": float(train_loss)})

        scheduler.step()

    print("Training completed!")

    best_metrics = {
        'spot_metrics': final_spot_metrics,
        'slide_avg_metrics': {k: float(v) for k, v in final_avg_slide_metrics.items()},
    }

    if logger:
        logger.log_best(best_metrics)
        logger.finish()

    return best_metrics, model
