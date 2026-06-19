import os
from pathlib import Path
from typing import Tuple

import hydra
from omegaconf import DictConfig

from src.span.preprocessing import SPAN_Padder
from lib.utils.coord_aug import build_coord_aug_transform
from lib.utils.entrypoint import initialize_hydra_run
from lib.utils.data import split_datasets
from lib.utils.model_utils import format_token_init_types
from .dataset import HiSTDataset, custom_collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
from src.span.model import create_block, SPAN_Encoder, SPAN_Decoder
from src.span.builders import build_model_config
from .trainer import train_gene_prediction

def get_run_name(cfg: DictConfig) -> str:
    return cfg.dataset_group if cfg.dataset_group is not None else cfg.dataset

def prepare_datasets(cfg: DictConfig) -> Tuple:

    if cfg.dataset_group is not None:
        dataset_names = cfg.dataset_groups[cfg.dataset_group]
    else:
        dataset_names = [cfg.dataset]

    train_slide_items = []
    test_slide_items = []

    for dataset_name in dataset_names:
        if "datasets" not in cfg or dataset_name not in cfg.datasets:
            available = list(getattr(cfg, "datasets", {}).keys())
            preview = ", ".join(available[:20]) + (" ..." if len(available) > 20 else "")
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Check `configs/dataset_config/*.yaml` or override via `dataset=<name>`. "
                f"Available (first {min(len(available), 20)}): {preview}"
            )

        dataset_cfg = cfg.datasets[dataset_name]
        src_folder = Path(dataset_cfg.path)
        gene_type = dataset_cfg.get('gene_type', None)
        num_genes = dataset_cfg.get('num_genes', None)

        if not src_folder.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist for '{dataset_name}': {src_folder}. "
                f"Override `datasets.{dataset_name}.path=...` or choose another dataset."
            )

        gene_dir_value = dataset_cfg.get('gene_dir', 'gene')
        gene_dir = Path(gene_dir_value)
        if not gene_dir.is_absolute():
            gene_dir = src_folder / gene_dir_value
        if not gene_dir.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' is missing required folder: {gene_dir}. "
                f"Expected per-slide files under `{gene_dir}/`."
            )

        slide_names = os.listdir(gene_dir)
        slide_names = [os.path.splitext(f)[0] for f in slide_names]
        train_slides, test_slides = split_datasets(
            slide_names, test_fold=cfg.test_fold, num_folds=4
        )

        gene_dir_str = str(gene_dir)
        for slide in train_slides:
            train_slide_items.append((str(src_folder), slide, gene_type, num_genes, gene_dir_str))
        for slide in test_slides:
            test_slide_items.append((str(src_folder), slide, gene_type, num_genes, gene_dir_str))

    if cfg.model.train_mode in ['frozen', 'lora', 'full']:
        image_suffix = 'RAW'
    else:
        image_suffix = cfg.patch_backbone

    apply_transform = build_coord_aug_transform(cfg)

    train_dataset = HiSTDataset(
        train_slide_items,
        transform=apply_transform,
        image_suffix=image_suffix
    )
    test_dataset = HiSTDataset(
        test_slide_items,
        image_suffix=image_suffix
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=cfg.training.num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    num_genes = cfg.datasets[dataset_names[0]].get('num_genes', None)
    if num_genes is None:
        sample = train_dataset[0]
        num_genes = sample.mol_feats.shape[1] if hasattr(sample, 'mol_feats') else None

    return train_loader, test_loader, num_genes, image_suffix

def build_model_components(cfg: DictConfig, num_genes: int):

    in_channels = 1280 if cfg.patch_backbone == 'V2' else 1024

    encoder_config, decoder_config, model_components = build_model_config(
        cfg,
        num_genes=num_genes,
        in_channels=in_channels,
        enc_act=cfg.model.enc_act
    )

    ds_layers = model_components['ds_layers']
    dilation = model_components['dilation']
    in_channels = model_components['in_channels']
    token_init_types = model_components['token_init_types']

    padder = SPAN_Padder(
        kernel_size=2, stride=2, dilation=dilation,
        n_layers=ds_layers, pad_feats=None
    )
    encoder_blocks = [
        create_block(config, i + 1, mask_padding=True)
        for i, config in enumerate(encoder_config)
    ]
    decoder_blocks = [
        create_block(config, len(decoder_config) - i)
        for i, config in enumerate(decoder_config)
    ]

    slide_encoder = SPAN_Encoder(
        encoder_blocks,
        embed_dim=in_channels,
        token_init_types=token_init_types
    )
    decoder = SPAN_Decoder(decoder_blocks)
    segmentor = nn.Identity()

    return encoder_config, decoder_config, model_components, padder, slide_encoder, decoder, segmentor

@hydra.main(version_base=None, config_path="../../configs", config_name="gene_prediction")
def main(cfg: DictConfig) -> None:
    cfg = initialize_hydra_run(
        cfg,
        "HiST Gene Prediction - Hydra Configuration",
        seed=0,
        drop_keys=("datasets", "dataset_groups"),
    )

    print("Preparing datasets...")
    train_loader, test_loader, num_genes, image_suffix = prepare_datasets(cfg)
    run_name = get_run_name(cfg)
    print(f"Dataset selection: {run_name}, Num genes: {num_genes}, Image suffix: {image_suffix}")


    print("Building model components...")
    (encoder_config, decoder_config, model_components,
     padder, slide_encoder, decoder, segmentor) = build_model_components(cfg, num_genes)

    gene_transform = lambda x: x

    model_config = build_model_config_string(cfg)

    print("Starting training...")
    best_metrics, model = train_gene_prediction(
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        image_suffix=image_suffix,
        gene_transform=gene_transform,
        padder=padder,
        slide_encoder=slide_encoder,
        decoder=decoder,
        segmentor=segmentor,
        model_config=model_config,
    )

    print("Saving results...")
    save_results_to_json(cfg, best_metrics)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60 + "\n")

def build_model_config_string(cfg: DictConfig) -> str:
    token_init_str = format_token_init_types(cfg.model.get('token_init_types', ['fix1e-4']))
    enc_act = cfg.model.get('enc_act', None)
    enc_act_str = "none" if enc_act is None else str(enc_act)
    input_proj_act = cfg.model.input_projection.get('activation', None)
    input_proj_act_str = "none" if input_proj_act is None else str(input_proj_act)
    return (
        f"{cfg.model.slide_configs}_"
        f"cf{cfg.model.channel_factor}_"
        f"so{cfg.model.skip_first}_"
        f"{cfg.model.trans_type}_"
        f"{cfg.model.econvs_type}_"
        f"{enc_act_str}_{input_proj_act_str}_"
        f"{cfg.patch_backbone}_"
        f"{cfg.model.global_strategy}_"
        f"{token_init_str}"
    )

def save_results_to_json(cfg: DictConfig, best_metrics: dict) -> None:
    from lib.utils.logging import update_results_file

    eval_metrics = cfg.training.eval_metrics

    results_dir = cfg.logging.results.dir
    save_path = cfg.logging.results.save_path

    model_config = build_model_config_string(cfg)
    run_name = get_run_name(cfg)
    idx = cfg.test_fold - 1

    def _update(results: dict) -> None:
        if run_name not in results:
            results[run_name] = {}

        if model_config not in results[run_name]:
            results[run_name][model_config] = {
                'spot_metrics': {metric: [None, None, None, None] for metric in eval_metrics},
                'slide_avg_metrics': {metric: [None, None, None, None] for metric in eval_metrics},
            }

        for metric_type in ['spot_metrics', 'slide_avg_metrics']:
            for metric_name, metric_value in best_metrics.get(metric_type, {}).items():
                if metric_name not in results[run_name][model_config][metric_type]:
                    results[run_name][model_config][metric_type][metric_name] = [None, None, None, None]
                results[run_name][model_config][metric_type][metric_name][idx] = metric_value

    update_results_file(results_dir, save_path, _update)

if __name__ == "__main__":
    main()
