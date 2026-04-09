# ProxyEval-Darts: Zero-Cost Proxy Evaluation on NAS-301 / EvoxBench-DARTS

Evaluates zero-cost proxies across DARTS architectures from NAS-Bench-301 or EvoxBench, computing Spearman correlation with ground-truth accuracy.

## Requirements

- **Python** 3.10
- **PyTorch** 2.0+ (with CUDA 11.8)
- **torchvision** 0.15+
- **numpy**, **scipy**, **tqdm**
- **evoxbench** (API mode requires a separate Python 3.7 env with evoxbench; file mode does not need it)

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

### EvoxBench Environment (API mode, separate)

API mode requires `evoxbench` for accuracy prediction. Since `evoxbench` depends on Python 3.7 and has specific dependency constraints, it is **incompatible** with the main PyTorch 3.10 environment.

**How we handle this:** We use a **cross-environment import** approach. `evoxbench` is installed in a separate conda environment, and at runtime the main PyTorch environment dynamically adds its `site-packages` to `sys.path`, allowing `evoxbench` to be imported without installing it in the main environment.

**EvoxBench environment:**

| Component | Version |
|-----------|--------|
| Python | 3.7.0 |
| evoxbench | 1.0.3 |

**Setup:**
```bash
# 1. Create separate evoxbench environment
conda create -n evoxbench python=3.7
conda activate evoxbench
pip install evoxbench
```

**Cross-environment import mechanism:**

In `eval_proxy.py`, the environment variable `EVOXBENCH_SITE_PACKAGES` points to the evoxbench environment's `site-packages` directory. At runtime, this path is appended to `sys.path`, enabling the main PyTorch process to import `evoxbench` directly:

```python
# Key code in eval_proxy.py
if os.environ.get('EVOXBENCH_SITE_PACKAGES'):
    sys.path.append(os.environ['EVOXBENCH_SITE_PACKAGES'])
```

**Set the environment variable before running:**
```bash
export EVOXBENCH_SITE_PACKAGES=/path/to/evoxbench_env/lib/python3.7/site-packages
```

## Data Loading Modes

### Mode 1: File Mode (default, recommended)
Load pre-computed architecture data from `.pkl` files. The data is included in `data/` — no additional download needed. File mode is provided because the original NAS-Bench-301 surrogate model and EvoxBench database are too large to distribute with this repository.

**Included files** (`data/`):
- `archs.pkl` — list of 2000 DARTS Genotype objects
- `nas301_indi.pkl` — NAS-Bench-301 accuracy indicators
- `exo_bench_indi.pkl` — EvoxBench accuracy indicators

### Mode 2: API Mode
Sample random architectures and query accuracy via the EvoxBench `DARTSBenchmark` API.
This requires `evoxbench` and its database configured.

- `--bench nas301` : uses the **test accuracy predictor** (deterministic)
- `--bench evoxbench` : uses the **validation accuracy predictor** (with noise)

## Supported Proxies

`MLFE`, `ES`, `synflow`, `meco`, `zico`, `zen`, `naswot`, `wrcor`, `epads`, `near`, `dextr`, `epsinas`, `swap`

## Usage

### File Mode — NAS-301 (500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --bench nas301 --num_archs 500
```

### File Mode — EvoxBench-DARTS (500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --bench evoxbench --num_archs 500
```

### API Mode — NAS-301 (500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode api --bench nas301 --num_archs 500
```

### API Mode — EvoxBench (500 random architectures)
```bash
python -u eval_proxy.py --proxy MLFE --mode api --bench evoxbench --num_archs 500
```

### All architectures (file mode, 2000 total)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --bench nas301
```

### Multi-Seed Run (30 seeds)
```bash
python -u eval_proxy.py --proxy MLFE --mode file --bench nas301 --num_runs 30
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--proxy` | Proxy name (required) |
| `--mode` | `file` or `api` |
| `--bench` | `nas301` (test acc, deterministic) or `evoxbench` (valid acc, with noise) |
| `--arc_data_path` | Path to directory with `.pkl` files (file mode) |
| `--data_root` | Root directory for CIFAR-10 (auto-downloaded if absent) |
| `--num_archs` | Number of architectures (random sample; omit for all) |
| `--batch_size` | Training batch size |
| `--seed` | Random seed |
| `--num_runs` | Number of runs with different seeds |
| `--output` | Output directory |

## Output

By default, results are saved to `../results/ProxyEval-Darts/<NAS301|EVOXBENCH>/cifar10/<proxy>/`:
- `proxy_seed_<seed>.npy` — proxy values per architecture
- `acc_seed_<seed>.npy` — ground-truth accuracies
- `time_seed_<seed>.npy` — computation time per architecture

Top-1 genotype is saved as `top1_genotype_<proxy>_<bench>_seed_<seed>.pkl`.

## Training

```bash
# CIFAR-10
python train_cifar.py --genotype_path ./top1_genotype_MLFE_nas301_seed_0.pkl

# ImageNet
python train_imagenet.py --genotype_path ./top1_genotype_MLFE_nas301_seed_0.pkl --data /path/to/imagenet
```

## Project Structure

```
ProxyEval-Darts/
├── eval_proxy.py        # Proxy evaluation script
├── train_cifar.py       # CIFAR-10 training (from searched genotype)
├── train_imagenet.py    # ImageNet training (from searched genotype)
├── data/                # Pre-computed architecture data (2000 architectures)
│   ├── archs.pkl
│   ├── nas301_indi.pkl
│   └── exo_bench_indi.pkl
├── cnn/                 # DARTS model & genotypes (shared, at ../cnn/)
├── proxieslib/          # Zero-cost proxy implementations (shared, at ../proxieslib/)
├── requirements.txt
└── README.md
```

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-301** (AutoML Freiburg) — DARTS surrogate benchmark
- **EvoxBench** (EMI Group) — Evolutionary multi-objective NAS benchmark
- **DARTS** (Liu et al.) — Differentiable architecture search space and CNN model
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework (`pruners/` module)

We gratefully acknowledge these projects for providing the foundation on which this work is built.
