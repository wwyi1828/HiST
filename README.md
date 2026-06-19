# HiST (Updating)

HiST is a research codebase for predicting spatial gene expression from histology
whole-slide images. Given patch-level image features and their spatial coordinates.

## Repository layout

```
configs/    Hydra configs (task, model, dataset, and logging)
lib/        Backbones and shared utilities
src/span/   SPAN encoder/decoder architecture, layers, and preprocessing
tasks/      Gene-prediction data, model, trainer, and entrypoint
```

## Installation

```bash
pip install -r requirements.txt
# or, for an editable install:
pip install -e .
```

## Data layout

Datasets are not bundled with this repository. Point the code at your data with
the `HIST_DATA_ROOT` environment variable (defaults to `./data`):

```bash
export HIST_DATA_ROOT=/path/to/datasets
```

Dataset directories are resolved relative to this root. The configs under
`configs/dataset_config/` expect subdirectories such as `ALLVisium/<dataset>`,
`Xenium_lung`, and `SPA_breast`.

Additional optional environment variables:

- `HIST_CKPT_ROOT` — directory holding pretrained slide-encoder checkpoints
  (defaults to `./checkpoints`).
- `UNI_WEIGHTS` — path to the [UNI](https://huggingface.co/MahmoodLab/UNI)
  patch-encoder weights, used when the `UNI` backbone is selected.

## Running

Training is a Hydra entrypoint. Override any config value on the command line.

```bash
# Train on a single dataset / fold
python -m tasks.gene_prediction.main dataset=NCBI_brain test_fold=1

# Swap the dataset group (configs/dataset_config/*.yaml)
python -m tasks.gene_prediction.main dataset_config=xenium dataset=Xenium_lung
```

## License

Released under the [MIT License](LICENSE).