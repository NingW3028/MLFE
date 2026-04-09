# ProxyEval-NAS201: Zero-Cost Proxy Evaluation on NAS-Bench-201

Evaluates zero-cost proxies across all 15,625 architectures in NAS-Bench-201 on three datasets, computing Spearman correlation with ground-truth test accuracy.

## Requirements

- **Python** 3.10
- **PyTorch** 2.0+ (with CUDA 11.8)
- **torchvision** 0.15+
- **numpy**, **scipy**
- **NAS-Bench-201 API**

### Tested Environment
| Component | Version |
|-----------|--------|
| Python | 3.10.10 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| torchvision | 0.15.1+cu118 |
| numpy | 1.26.4 |
| scipy | 1.10.1 |

Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Prerequisites

- **NAS-Bench-201 API file** (`.pth`), e.g. `NAS-Bench-201-v1_0-e61699.pth`
- **CIFAR-10** / **CIFAR-100**: Auto-downloaded into `--data_root`
- **ImageNet16-120**: Auto-downloaded via `gdown` (install: `pip install gdown`). If auto-download fails, manually download from [Google Drive](https://drive.google.com/drive/folders/1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40) and place at `<data_root>/ImageNet16/`

## Supported Proxies

`MLFE`, `ES`, `synflow`, `meco`, `zico`, `zen`, `naswot`, `wrcor`, `epads`, `near`, `dextr`, `epsinas`, `swap`

## Usage

### Single Run (ImageNet16-120, default)
```bash
python -u eval_proxy.py --proxy MLFE \
  --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth \
  --data_root /path/to/data
```

### Evaluate on All Datasets
```bash
python -u eval_proxy.py --proxy MLFE \
  --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth \
  --data_root /path/to/data \
  --datasets cifar10 cifar100 ImageNet16-120
```

### Multi-Seed Run (30 seeds)
```bash
python -u eval_proxy.py --proxy MLFE \
  --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth \
  --data_root /path/to/data \
  --num_runs 30 --seed 42
```

> **Note:** `--data_root` should point to a directory containing an `ImageNet16/` subfolder (with `train_data_batch_*` and `val_data` files). CIFAR-10/100 will be auto-downloaded into the same directory.

### Arguments

| Argument | Description |
|----------|-------------|
| `--proxy` | Proxy name (required) |
| `--api_path` | Path to NAS-Bench-201 API file (required) |
| `--datasets` | Datasets to evaluate on |
| `--data_root` | Root directory for datasets |
| `--batch_size` | Training batch size |
| `--seed` | Random seed |
| `--num_runs` | Number of runs with different seeds |
| `--output` | Output directory |

## Output

By default, results are saved to `../results/ProxyEval-NAS201/NAS201/<dataset>/<proxy>/`:
- `proxy_seed_<seed>.npy` — proxy values per architecture
- `acc_seed_<seed>.npy` — ground-truth accuracies
- `time_seed_<seed>.npy` — computation time per architecture

## Project Structure

```
ProxyEval-NAS201/
├── eval_proxy.py      # Main evaluation script
├── models/            # NAS-201 cell-based tiny net
├── proxieslib/        # Zero-cost proxy implementations (MLFE, synflow, etc.)
├── pruners/           # Additional proxy measures (snip, grasp, etc.)
├── utilsfold/         # ImageNet16 dataset loader
├── requirements.txt
└── README.md
```

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-201** (Dong & Yang) — NAS benchmark dataset and API
- **XAutoDL / AutoDL-Projects** (Dong & Yang) — Cell-based model implementations and utilities
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework (`pruners/` module)

We gratefully acknowledge these projects for providing the foundation on which this work is built.
