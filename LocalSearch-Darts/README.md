# NAS-301 Local Search - Standalone Version

MacroNAS Local Search implementation for NAS-Bench-301 (DARTS) search space.

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

### EvoxBench Environment (optional, separate)

For accuracy prediction via EvoxBench, `evoxbench` is required. Since it depends on Python 3.7 and has specific dependency constraints, it is **incompatible** with the main PyTorch 3.10 environment.

**How we handle this:** We use a **cross-environment import** approach. `evoxbench` is installed in a separate conda environment, and at runtime the main PyTorch environment dynamically adds its `site-packages` to `sys.path`, allowing `evoxbench` to be imported without installing it in the main environment.

**EvoxBench environment:**

| Component | Version |
|-----------|--------|
| Python | 3.7.0 |
| evoxbench | 1.0.3 |

**Setup:**
```bash
conda create -n evoxbench python=3.7
conda activate evoxbench
pip install evoxbench
```

**Cross-environment import mechanism:**

In `301_macronas_local_search.py`, the environment variable `EVOXBENCH_SITE_PACKAGES` points to the evoxbench environment's `site-packages` directory. At runtime, this path is appended to `sys.path`:

```python
# Key code in 301_macronas_local_search.py
if os.environ.get('EVOXBENCH_SITE_PACKAGES'):
    sys.path.append(os.environ['EVOXBENCH_SITE_PACKAGES'])
```

**Set the environment variable before running:**
```bash
export EVOXBENCH_SITE_PACKAGES=/path/to/evoxbench_env/lib/python3.7/site-packages
```

## Dataset Setup

Download CIFAR-10 dataset and place it in `data/cifar10` directory:
```bash
mkdir -p data/cifar10
# Dataset will be auto-downloaded on first run
```

## Surrogate Model Setup (Optional)

To enable accuracy prediction with `--use_benchmark`:

1. Install nasbench301:
```bash
pip install nasbench301
```

2. Download surrogate model (gnn_gin_v0.9) from:
   - nasbench301 surrogate model

3. Place model in `nb_models_0.9/gnn_gin_v0.9/` directory:
```bash
mkdir -p nb_models_0.9
# Download and extract model files to nb_models_0.9/gnn_gin_v0.9/
```

**Note**: Surrogate model is ~3.7GB, not included in this repository.

## Usage

### Single Proxy Test
```bash
python run_search_proxy.py --proxy MLFE --max_evals 100
```

### With Accuracy Query (NAS-301)
```bash
python run_search_proxy.py --proxy MLFE --max_evals 100 --bench nas301
```

### With Accuracy Query (EvoxBench)
```bash
python run_search_proxy.py --proxy MLFE --max_evals 100 --bench evoxbench
```

Arguments:
- `--proxy`: Proxy name (default: `MLFE`)
- `--max_evals`: Maximum evaluations (default: `100`)
- `--bench`: Query accuracy after search — `nas301` (test acc, deterministic) or `evoxbench` (valid acc, with noise). Requires evoxbench env.

### Batch Run All Proxies
```bash
python run_all_proxies_multi_runs.py
```
This runs 30 experiments per proxy with different seeds.

### Search with Accuracy Prediction
```bash
python 301_macronas_local_search.py --proxy naswot --max_evals 500 --num_searches 10 --output results.pkl --use_benchmark
```

Arguments:
- `--proxy`: Proxy name (required)
- `--max_evals`: Maximum evaluations per search (default: 500)
- `--num_searches`: Number of local searches (default: 10)
- `--seed`: Random seed (default: 42)
- `--output`: Output pickle file (required)
- `--use_benchmark`: Enable NAS-Bench-301 accuracy prediction (optional)

## Files

| File | Description |
|------|-------------|
| `local_search_proxy.py` | Core Local Search implementation |
| `run_search_proxy.py` | Single proxy test script |
| `run_all_proxies_multi_runs.py` | Batch run script (30 runs per proxy) |
| `301_macronas_local_search.py` | Search with accuracy prediction |
| `xautodl/` | DARTS network implementation |
| `naslib/` | NASLib search space library |
| `pruners/` | Proxy implementations |
| `proxieslib/` | Additional proxy implementations |
| `requirements.txt` | Python dependencies |

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

By default, results are saved to `../results/LocalSearch-Darts/` as pickle/JSON files containing:
- Best genotype
- Best proxy score
- Best predicted accuracy (if `--bench` specified)
- Best parameters count
- All search results

## Acknowledgments

This project builds upon the following open-source works:

- **NAS-Bench-301** (AutoML Freiburg) — DARTS surrogate benchmark
- **EvoxBench** (EMI Group) — Evolutionary multi-objective NAS benchmark
- **DARTS** (Liu et al.) — Differentiable architecture search space and CNN model
- **XAutoDL / AutoDL-Projects** (Dong & Yang) — DARTS network implementation
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework and NAS search space library

We gratefully acknowledge these projects for providing the foundation on which this work is built.
