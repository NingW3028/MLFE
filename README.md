# Neural Architecture Selection via Maximizing Local Entropy of Feature Maps

> [!IMPORTANT]
> **Errata:** The aggregation formula `v = ln(max(Δ)) · Σ(r)` shown in the manuscript is an early version. All experiments in the paper were conducted using the updated formula:
>
> **`v = max(Δ) · Σ(r)`**
>
> The formula in the manuscript was not updated accordingly. **The code in this repository reflects the correct version used in all experiments.** This change does not affect any conclusions or experimental results reported in the paper. The formula in the manuscript will be corrected in the next revision.

This repository provides the code for evaluating and searching neural architectures using zero-cost proxies, including our proposed MLFE (Mixed-Granularity Local Feature Entropy) proxy. The project covers three NAS benchmarks — NAS-Bench-101, NAS-Bench-201, and DARTS (NAS-Bench-301 / EvoxBench) — and extends to Vision Transformer search spaces via AutoFormer.

The codebase is organized into multiple subprojects, each targeting a specific benchmark and task (proxy evaluation or proxy-guided local search). Every subproject contains its own detailed README with full usage instructions, environment setup, and data preparation guides.

## Subprojects Overview

### Local Search (LocalSearch)

Use zero-cost proxies as the fitness function to guide local search for high-performing architectures without training.

| Subproject | Benchmark | Search Method | README |
|---|---|---|---|
| **LocalSearch-nas101** | NAS-Bench-101 | Graph-based mutation | [README](LocalSearch-nas101/README.md) |
| **LocalSearch-nas201** | NAS-Bench-201 | Operation-level mutation | [README](LocalSearch-nas201/README.md) |
| **LocalSearch-Darts** | NAS-Bench-301 / EvoxBench | DARTS genotype mutation | [README](LocalSearch-Darts/README.md) |

### Proxy Evaluation (ProxyEval)

Evaluate the Spearman rank correlation between zero-cost proxies and ground-truth accuracy across sampled architectures. ProxyEval-NAS101 and ProxyEval-Darts provide a **file mode** (default) with pre-computed architecture data, because the original benchmark API files (NAS-Bench-101 `.tfrecord` ~499 MB, NAS-Bench-301 surrogate model, EvoxBench database) are too large to distribute with this repository.

| Subproject | Benchmark | Datasets | README |
|---|---|---|---|
| **ProxyEval-NAS101** | NAS-Bench-101 | CIFAR-10 | [README](ProxyEval-NAS101/README.md) |
| **ProxyEval-NAS201** | NAS-Bench-201 | CIFAR-10/100, ImageNet16-120 | [README](ProxyEval-NAS201/README.md) |
| **ProxyEval-Darts** | NAS-Bench-301 / EvoxBench | CIFAR-10 | [README](ProxyEval-Darts/README.md) |
| **ProxyEval-Trans101** | TransNAS-Bench-101 | Taskonomy (7 tasks) | [README](ProxyEval-Trans101/README.md) |

### AutoFormer (Vision Transformer)

Evaluate zero-cost proxies on Vision Transformer architectures using the AutoFormer supernet, with optional subnet validation and fine-tuning on ImageNet.

| Subproject | Search Space | README |
|---|---|---|
| **autoformer** | AutoFormer-Tiny / Small / Base | [README](autoformer/README.md) |

## Environment

This project integrates multiple public NAS benchmarks (NAS-Bench-101, NAS-Bench-201, NAS-Bench-301, EvoxBench) as evaluation backends. These benchmarks have mutually incompatible dependencies (e.g., NAS-Bench-101 requires TensorFlow 1.15 / Python 3.7, while the main codebase uses PyTorch 2.0 / Python 3.10). To address this, we provide pre-computed file modes and cross-environment imports, so that most experiments can run in a single PyTorch environment without installing conflicting packages. Each subproject includes its own README with dedicated environment configuration and `requirements.txt`, allowing independent setup per benchmark.

### Base Environment

All subprojects share the following base environment:

| Component | Version |
|-----------|---------|
| Python | 3.10.10 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| torchvision | 0.15.1+cu118 |
| numpy | 1.26.4 |
| scipy | 1.10.1 |

### Important: Per-Subproject Environment Differences

> **⚠ Different subprojects may require different environments due to benchmark API incompatibilities.** The underlying NAS benchmarks have conflicting dependencies (e.g., TensorFlow 1.15 requires Python 3.7 while the main environment uses Python 3.10). Please read the README of each subproject carefully before running it.

Key environment notes:

- **ProxyEval-NAS101** (API mode): Requires a **separate conda environment** with Python 3.7 + TensorFlow 1.15 for querying the NAS-Bench-101 API directly. The main environment uses a cross-environment import to access TensorFlow. File mode (default) does not require TensorFlow.
- **ProxyEval-Darts / LocalSearch-Darts** (EvoxBench mode): Requires a **separate conda environment** with Python 3.7 + evoxbench 1.0.3 for EvoxBench accuracy prediction. The main environment uses a cross-environment import. NAS-Bench-301 mode (default) does not require this.
- **autoformer**: Requires `timm==0.9.2` and ImageNet-1K dataset + pre-trained AutoFormer supernet checkpoints.
- **Other subprojects**: Only need the base environment (Python 3.10 + PyTorch 2.0).

Each subproject provides a `requirements.txt` (where applicable) for easy dependency installation.

## How to Run

### General Workflow

1. **Choose a subproject** based on your target benchmark and task.
2. **Read the subproject's README** for specific environment setup, data preparation, and usage instructions.
3. **Install dependencies** in the subproject directory.
4. **Run the main script** with the desired proxy and arguments.
5. **Collect results** from the unified `results/` directory.

### Quick Start Examples

Below are the commands used to reproduce all experiments in this project.

#### 1. LocalSearch — Proxy-Guided Architecture Search

**LocalSearch-NAS101** (accuracy auto-queried from `nas101_cache.pkl`)
```bash
cd LocalSearch-nas101
python run_search_proxy.py --proxy MLFE --max_evals 100
```

**LocalSearch-NAS201** (with optional NAS-Bench-201 accuracy query)
```bash
cd LocalSearch-nas201
# Search only (no API)
python run_search_proxy.py --proxy MLFE --dataset cifar10 --max_evals 100
# Search + accuracy query
python run_search_proxy.py --proxy MLFE --dataset cifar10 --max_evals 100 \
    --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth
```

**LocalSearch-Darts** (with optional NAS-301 / EvoxBench accuracy query)
```bash
cd LocalSearch-Darts
# Search only (no API)
python run_search_proxy.py --proxy MLFE --max_evals 100
# Search + NAS-301 accuracy (requires evoxbench env)
python run_search_proxy.py --proxy MLFE --max_evals 100 --bench nas301
# Search + EvoxBench accuracy
python run_search_proxy.py --proxy MLFE --max_evals 100 --bench evoxbench
```

> For NAS-301 / EvoxBench accuracy query, set `EVOXBENCH_SITE_PACKAGES` to your evoxbench conda env's site-packages path before running.

#### 2. ProxyEval — Proxy Evaluation (Spearman Correlation)

**ProxyEval-NAS101** — File Mode (pre-computed data, no TensorFlow needed)
```bash
cd ProxyEval-NAS101
python -u eval_proxy.py --proxy MLFE --mode file --num_archs 500
```

**ProxyEval-NAS101** — API Mode (requires TF 1.15 env, set `TF_SITE_PACKAGES`)
```bash
cd ProxyEval-NAS101
python -u eval_proxy.py --proxy MLFE --mode api --api_path /path/to/nasbench_only108.tfrecord --num_archs 500
```

**ProxyEval-NAS201** (requires NAS-Bench-201 API file)
```bash
cd ProxyEval-NAS201
python -u eval_proxy.py --proxy MLFE --api_path /path/to/NAS-Bench-201-v1_0-e61699.pth
```

**ProxyEval-Darts** — File Mode (pre-computed data, no EvoxBench env needed)
```bash
cd ProxyEval-Darts
# NAS-301
python -u eval_proxy.py --proxy MLFE --mode file --bench nas301 --num_archs 500
# EvoxBench
python -u eval_proxy.py --proxy MLFE --mode file --bench evoxbench --num_archs 500
```

**ProxyEval-Darts** — API Mode (requires evoxbench env, set `EVOXBENCH_SITE_PACKAGES`)
```bash
cd ProxyEval-Darts
# NAS-301
python -u eval_proxy.py --proxy MLFE --mode api --bench nas301 --num_archs 500
# EvoxBench
python -u eval_proxy.py --proxy MLFE --mode api --bench evoxbench --num_archs 500

# Train top-1 architecture on CIFAR-10
python train_cifar.py --genotype_path ./top1_genotype_MLFE_nas301_seed_0.pkl
# Train top-1 architecture on ImageNet
python train_imagenet.py --genotype_path ./top1_genotype_MLFE_nas301_seed_0.pkl --data /path/to/imagenet
```

**ProxyEval-Trans101** (requires TransNAS-Bench-101 API file)
```bash
cd ProxyEval-Trans101
python eval_proxy.py --api_path /path/to/transnas-bench_v10141024.pth --data_dir /path/to/taskonomy_images
```

#### 3. AutoFormer — Vision Transformer Proxy Evaluation

**Proxy evaluation only** (no fine-tuning)
```bash
cd autoformer
# Tiny space
python random_sample_eval.py --proxy MLFE --space tiny --num_samples 100 \
    --data_root /path/to/ImageNet1K --checkpoint /path/to/supernet-tiny.pth

# Small space
python random_sample_eval.py --proxy MLFE --space small --num_samples 100 \
    --data_root /path/to/ImageNet1K --checkpoint /path/to/supernet-small.pth

# Base space
python random_sample_eval.py --proxy MLFE --space base --num_samples 100 \
    --data_root /path/to/ImageNet1K --checkpoint /path/to/supernet-base.pth
```

**With fine-tuning** (fine-tune the best architecture found by proxy)
```bash
cd autoformer
python random_sample_eval.py --proxy MLFE --space tiny --num_samples 100 \
    --data_root /path/to/ImageNet1K --checkpoint /path/to/supernet-tiny.pth \
    --finetune_epochs 40 --finetune_lr 5e-4 --finetune_batch_size 128
```

> **Tip:** For each subproject, refer to its README for the full list of arguments and advanced usage options (e.g., multi-seed runs, different benchmarks, fine-tuning).

### Ablation Studies

**Batch size ablation:** Configure via `--batch_size` in each eval script.

**Entropy variant ablation:** Replace `compute_MLFE` with `compute_MLFE_ablation` from `proxieslib/MLFE_ablation.py`, and set `entropy_mode` and `scale_mode`:

| Variant | `entropy_mode` | `scale_mode` | Description |
|---------|---------------|-------------|-------------|
| DE+GE | `'de'` | `'ge'` | Standard discrete entropy + global entropy |
| MGE+GE | `'mge'` | `'ge'` | Mixed-granularity entropy + global entropy |
| DE+LE | `'de'` | `'le'` | Standard discrete entropy + local entropy |
| MGE+LE | `'mge'` | `'le'` | Mixed-granularity entropy + local entropy (= MLFE) |

```python
# Example: switch from MLFE to DE+GE in eval script
from proxieslib.MLFE_ablation import compute_MLFE_ablation
score = compute_MLFE_ablation(network, x, dtype, entropy_mode='de', scale_mode='ge', benchtype='Darts')
```

## Acknowledgments

This project builds upon several excellent open-source works:

- **NAS-Bench-101** (Google Research) — NAS-Bench-101 benchmark dataset and API
- **nasbench-pytorch** — PyTorch implementation of NAS-Bench-101 models
- **NAS-Bench-201** (Dong & Yang) — NAS-Bench-201 benchmark dataset and API
- **XAutoDL / AutoDL-Projects** (Dong & Yang) — Cell-based model implementations and utilities
- **TransNAS-Bench-101** (Noah's Ark Lab, Huawei) — TransNAS-Bench-101 benchmark for transfer learning NAS
- **NAS-Bench-301** (AutoML Freiburg) — DARTS surrogate benchmark
- **EvoxBench** (EMI Group) — Evolutionary multi-objective NAS benchmark
- **DARTS** (Liu et al.) — Differentiable architecture search
- **NASLib** (Samsung / AutoML Freiburg) — Zero-cost proxy framework
- **AutoFormer / Cream** (Microsoft) — AutoFormer supernet and Vision Transformer search space

We gratefully acknowledge these projects for providing the foundation on which this work is built.

## License

This project is licensed under the Apache License 2.0. See individual source files for details.
