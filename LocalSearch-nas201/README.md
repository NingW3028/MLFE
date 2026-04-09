# NAS-201 Local Search - Standalone Version

MacroNAS Local Search implementation for NAS-Bench-201 search space.

## Requirements
- Python 3.10
- PyTorch 2.0+ (with CUDA 11.8)
- torchvision 0.15+
- CUDA-capable GPU (recommended)

### Tested Environment
| Component | Version |
|-----------|--------|
| Python | 3.10.10 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| torchvision | 0.15.1+cu118 |
| numpy | 1.26.4 |
| scipy | 1.10.1 |

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### CIFAR-10
```bash
mkdir -p data/cifar10
# Dataset will be auto-downloaded on first run via run_search_proxy.py
```

### CIFAR-100
```bash
mkdir -p data/cifar100
# Dataset will be auto-downloaded on first run via run_search_proxy.py
```

### ImageNet16-120 (Optional)
Download the downsampled ImageNet16 dataset and place it in `data/ImageNet16/`:
```bash
mkdir -p data/ImageNet16
# Download from: https://arxiv.org/abs/1707.08819
# Place train_data_batch_1 ... train_data_batch_10 and val_data files
```

## NAS-Bench-201 API (Optional)

To enable ground truth accuracy query with `--api_path`:

1. Install the NAS-201 API:
```bash
pip install nas-bench-201
```

2. Download NAS-Bench-201 benchmark file:
   - `NAS-Bench-201-v1_0-e61699.pth`

3. Provide the path via `--api_path` argument.

**Note**: The API file is ~2GB, not included in this repository.

## Usage

### Single Proxy Test
```bash
python run_search_proxy.py --proxy MLFE --dataset cifar10 --max_evals 100
```

Arguments:
- `--proxy`: Proxy name (default: `MLFE`)
- `--dataset`: Dataset to use: `cifar10`, `cifar100`, `ImageNet16-120` (default: `cifar10`)
- `--max_evals`: Maximum evaluations (default: `100`)
- `--api_path`: Path to NAS-Bench-201 API file for ground-truth accuracy query (optional)

When `--api_path` is provided, ground-truth test accuracy is automatically queried for each found architecture after search completes.

### Batch Run All Proxies
```bash
python run_all_proxies_multi_runs.py
```
This runs 30 experiments per proxy with different seeds. Modify `DATASET` variable to change the dataset.

### Search with Accuracy Query
```bash
python 201_macronas_local_search.py --proxy naswot --dataset cifar10 --max_evals 500 --num_searches 20 --output results.pkl
```

With NAS-Bench-201 ground truth accuracy:
```bash
python 201_macronas_local_search.py --proxy MLFE --dataset cifar10 --max_evals 500 --num_searches 20 --output results.pkl --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth
```

Arguments:
- `--proxy`: Proxy name (required)
- `--dataset`: Dataset to use: `cifar10`, `cifar100`, `ImageNet16-120` (default: cifar10)
- `--max_evals`: Maximum evaluations per search (default: 500)
- `--num_searches`: Number of local searches (default: 20)
- `--seed`: Random seed (default: 42)
- `--output`: Output pickle file (required)
- `--data_root`: Root directory for datasets (default: data)
- `--api_path`: Path to NAS-Bench-201 API file (optional)

## Files

| File | Description |
|------|-------------|
| `local_search_proxy.py` | Core Local Search implementation for NAS-201 |
| `run_search_proxy.py` | Single proxy test script |
| `run_all_proxies_multi_runs.py` | Batch run script (30 runs per proxy) |
| `201_macronas_local_search.py` | Search with accuracy query |
| `models/` | NAS-Bench-201 network implementation |
| `utilsfold/` | Utility modules (ImageNet16 dataset loader) |
| `pruners/` | Proxy implementations |
| `proxieslib/` | Additional proxy implementations |

## NAS-201 Search Space

NAS-Bench-201 uses a cell-based search space with:
- **4 nodes** (1 input + 3 intermediate)
- **6 edges** connecting nodes
- **5 operations** per edge: `none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`
- Total: 5^6 = 15,625 unique architectures

Architecture string format: `|op~0|+|op~0|op~1|+|op~0|op~1|op~2|`

## Supported Proxies

| Proxy | Description |
|-------|-------------|
| ES | Entropy Score |
| synflow | Synaptic Flow |
| epsinas | EP-NAS proxy |
| naswot | NASWOT score |
| MLFE | Mixed-Granularity Local Feature Entropy |
| epads | EPADS proxy |
| wrcor | Weighted Correlation |
| zico | ZICO proxy |
| swap | SWAP proxy |
| dextr | DEXTR proxy |
| zen | Zen score |
| meco | MECO proxy |
| near | NEAR proxy |

## Supported Datasets

| Dataset | Classes | Image Size | Description |
|---------|---------|------------|-------------|
| cifar10 | 10 | 32x32 | CIFAR-10 |
| cifar100 | 100 | 32x32 | CIFAR-100 |
| ImageNet16-120 | 120 | 16x16 | Downsampled ImageNet (120 classes) |

## Results

By default, results are saved to `../results/LocalSearch-nas201/` as pickle/JSON files containing:
- Best genotype (list of 6 integers)
- Best architecture string
- Best proxy score
- Best parameters count
- Best accuracy (if API available)
- All search results

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-201** (Dong & Yang) — NAS benchmark dataset and API
- **XAutoDL / AutoDL-Projects** (Dong & Yang) — Cell-based model implementations and utilities
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework (`pruners/` module)

We gratefully acknowledge these projects for providing the foundation on which this work is built.
