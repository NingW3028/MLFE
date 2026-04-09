# ProxyEval-NAS101: Zero-Cost Proxy Evaluation on NAS-Bench-101

Evaluates zero-cost proxies across architectures in NAS-Bench-101, computing Spearman correlation with ground-truth test accuracy.

## Requirements

- **Python** 3.10
- **PyTorch** 2.0+ (with CUDA 11.8)
- **torchvision** 0.15+
- **numpy**, **scipy**, **tqdm**

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

### API Mode (optional, separate environment)

API mode queries the NAS-Bench-101 dataset directly via the `nasbench` API, which depends on **TensorFlow 1.15**. Since TensorFlow 1.15 requires Python 3.7, it is **incompatible** with the main PyTorch 3.10 environment.

**How we handle this:** We use a **cross-environment import** approach. TensorFlow is installed in a separate conda environment, and at runtime the main PyTorch environment dynamically adds the TensorFlow environment's `site-packages` to `sys.path`, allowing `tensorflow` and `nasbench` to be imported without installing them in the main environment.

**TensorFlow environment:**

| Component | Version |
|-----------|--------|
| Python | 3.7.16 |
| TensorFlow | 1.15.0 |
| nasbench | — |

**Setup:**
```bash
# 1. Create separate TensorFlow environment
conda create -n tf python=3.7
conda activate tf
pip install tensorflow==1.15.0
pip install nasbench
```

- NAS-Bench-101 dataset file `nasbench_only108.tfrecord`

**Cross-environment import mechanism:**

In `eval_proxy.py`, the environment variable `TF_SITE_PACKAGES` points to the TensorFlow environment's `site-packages` directory. At runtime, this path is appended to `sys.path`, enabling the main PyTorch process to import `tensorflow` and `nasbench` directly:

```python
# Key code in eval_proxy.py
if os.environ.get('TF_SITE_PACKAGES'):
    sys.path.append(os.environ['TF_SITE_PACKAGES'])
```

**Set the environment variable before running:**
```bash
export TF_SITE_PACKAGES=/path/to/tf_env/lib/python3.7/site-packages
```

## Data Loading Modes

### Mode 1: File Mode (default, recommended)
Load pre-computed architecture data from `.npy` files. The data is included in the `data/` directory — no additional download needed. File mode is provided because the original NAS-Bench-101 API file (`nasbench_only108.tfrecord`, ~499 MB) is too large to distribute with this repository.

**Included files** (`data/`):
- `keep_indi_all.npy` — indicator array (col 4 = test accuracy)
- `keep_arc_all.npy` — operation labels per architecture
- `keep_matrix_list_all.npy` — adjacency matrices

### Mode 2: API Mode
Query NAS-Bench-101 API directly from a `.tfrecord` file. Requires TensorFlow and `nasbench` (see above).

## Supported Proxies

`MLFE`, `ES`, `synflow`, `meco`, `zico`, `zen`, `naswot`, `wrcor`, `epads`, `near`, `dextr`, `epsinas`, `swap`, `snip`, `grasp`, `fisher`, `jacob_cov`, `grad_norm`, `ntk`

## Usage

### File Mode (pre-computed data, 500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --num_archs 500
```

### API Mode (query NAS-Bench-101 directly, 500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode api --api_path /path/to/nasbench_only108.tfrecord --num_archs 500
```

### All architectures (file mode, 2000 total)
```bash
python -u eval_proxy.py --proxy MLFE --mode file
```

### Multi-Seed Run (30 seeds)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --num_runs 30 --seed 42
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--proxy` | Proxy name (required) |
| `--mode` | `file` or `api` |
| `--arc_data_path` | Path to directory with `.npy` files (file mode) |
| `--api_path` | Path to `nasbench_only108.tfrecord` (required for api mode) |
| `--data_root` | Root directory for CIFAR-10 (auto-downloaded if absent) |
| `--num_archs` | Number of architectures (random sample; omit for all) |
| `--batch_size` | Training batch size |
| `--seed` | Random seed |
| `--num_runs` | Number of runs with different seeds |
| `--output` | Output directory |

## Output

By default, results are saved to `../results/ProxyEval-NAS101/NAS101/cifar10/<proxy>/`:
- `proxy_seed_<seed>.npy` — proxy values per architecture
- `acc_seed_<seed>.npy` — ground-truth accuracies
- `time_seed_<seed>.npy` — computation time per architecture

## Project Structure

```
ProxyEval-NAS101/
├── eval_proxy.py          # Main evaluation script
├── data/                  # Pre-computed architecture data (2000 architectures)
│   ├── keep_indi_all.npy
│   ├── keep_arc_all.npy
│   └── keep_matrix_list_all.npy
├── nasbench_pytorch/      # NAS-Bench-101 PyTorch model
├── proxieslib/            # Zero-cost proxy implementations (MLFE, synflow, etc.)
├── pruners/               # Additional proxy measures (snip, grasp, etc.)
├── requirements.txt
└── README.md
```

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-101** (Google Research) — NAS benchmark dataset and API
- **nasbench-pytorch** — PyTorch implementation of NAS-Bench-101 models
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework (`pruners/` module)

We gratefully acknowledge these projects for providing the foundation on which this work is built.
