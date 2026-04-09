# NAS-101 Local Search - Standalone Version

MacroNAS Local Search implementation for NAS-Bench-101 search space.

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

Download CIFAR-10 dataset and place it in `data/cifar10` directory:
```bash
mkdir -p data/cifar10
# Dataset will be auto-downloaded on first run, or manually download from:
# https://www.cs.toronto.edu/~kriz/cifar.html
```

## Usage

### Single Proxy Test
```bash
python run_search_proxy.py --proxy MLFE --max_evals 100
```

Arguments:
- `--proxy`: Proxy name (default: `MLFE`)
- `--max_evals`: Maximum evaluations (default: `100`)

After search completes, ground-truth accuracy is automatically queried from `nas101_cache.pkl` (no TensorFlow or external API needed).

### Batch Run All Proxies
```bash
python run_all_proxies_multi_runs.py
```
This runs 30 experiments per proxy with different seeds.

### Search with Accuracy Query
```bash
python 101_macronas_local_search.py --proxy naswot --max_evals 500 --num_searches 20 --output results.pkl
```

Arguments:
- `--proxy`: Proxy name (required)
- `--max_evals`: Maximum evaluations per search (default: 500)
- `--num_searches`: Number of local searches (default: 20)
- `--seed`: Random seed (default: 42)
- `--output`: Output pickle file (required)
- `--cache_path`: Path to NAS-101 cache (default: nas101_cache.pkl)

## Files

| File | Description |
|------|-------------|
| `local_search_proxy.py` | Core Local Search implementation |
| `run_search_proxy.py` | Single proxy test script |
| `run_all_proxies_multi_runs.py` | Batch run script (30 runs per proxy) |
| `101_macronas_local_search.py` | Search with accuracy query |
| `preload_data.py` | Preload NAS-Bench-101 data |
| `nas101_cache.pkl` | NAS-101 accuracy cache |
| `nasbench_pytorch/` | NAS-Bench-101 network implementation |
| `naslib/` | NASLib search space library |
| `pruners/` | Proxy implementations |
| `proxieslib/` | Additional proxy implementations |

## Benchmark Cache

`nas101_cache.pkl` contains test accuracy for all 423,624 architectures in NAS-Bench-101. The original NAS-Bench-101 API requires TensorFlow 1.15 and a large `.tfrecord` file (~499 MB). This cache file provides a PyTorch-compatible alternative that can be loaded directly via pickle, enabling quick accuracy lookup without TensorFlow or any external API file.

**Download:** [nas101_cache.pkl](https://pan.baidu.com/s/1wHDk6qPutBtD-xahM3e6SQ?pwd=a8b4)

After downloading, place the file in the project root directory:
```
LocalSearch-nas101/
└── nas101_cache.pkl
```

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

## Results

By default, results are saved to `../results/LocalSearch-nas101/` as pickle/JSON files containing:
- Best architecture (matrix and ops)
- Best proxy score
- Best accuracy (if cache available)
- All search results

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-101** (Google Research) — NAS benchmark dataset and API
- **nasbench-pytorch** — PyTorch implementation of NAS-Bench-101 models
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework (`pruners/` module)

We gratefully acknowledge these projects for providing the foundation on which this work is built.
