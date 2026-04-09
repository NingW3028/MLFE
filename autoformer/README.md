# AutoFormer Zero-Cost Proxy Evaluation

Zero-cost proxy evaluation for the AutoFormer (Vision Transformer) search space.
Randomly samples candidate architectures from the supernet, evaluates each with a zero-cost proxy,
selects the best, validates its accuracy on ImageNet, and optionally fine-tunes it.

## Requirements
- Python 3.10
- PyTorch 2.0+ (with CUDA 11.8)
- torchvision 0.15+
- timm 0.9.x
- CUDA-capable GPU (recommended)
- ImageNet-1K dataset
- Pre-trained AutoFormer supernet checkpoints

### Tested Environment
| Component | Version |
|-----------|--------|
| Python | 3.10.10 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| torchvision | 0.15.1+cu118 |
| timm | 0.9.2 |
| numpy | 1.26.4 |
| scipy | 1.10.1 |

> **Note**: `timm` version matters. AutoFormer supernet relies on `timm` APIs that may change across versions. `timm==0.9.2` is verified to work.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset & Checkpoint Setup

### ImageNet-1K
```
<data_root>/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### Pre-trained Supernet Checkpoints

Each search space requires the corresponding supernet checkpoint (`.pth`):

| Space | Checkpoint | Supernet Config |
|-------|-----------|-----------------|
| tiny  | `supernet-tiny.pth`  | embed=256, depth=14, heads=4 |
| small | `supernet-small.pth` | embed=448, depth=14, heads=7 |
| base  | `supernet-base.pth`  | embed=640, depth=16, heads=10 |

These checkpoints are from the AutoFormer project.
The `--checkpoint` argument is **required** to use pre-trained supernet weights for proxy evaluation and subnet validation.

## Usage

### Basic: Proxy Evaluation + Subnet Validation

Sample 100 random architectures, evaluate with MLFE proxy, validate the best subnet on ImageNet val:

```bash
python random_sample_eval.py \
    --proxy MLFE \
    --space tiny \
    --num_samples 100 \
    --seed 42 \
    --output results_tiny.pkl \
    --data_root /path/to/ImageNet1K \
    --checkpoint /path/to/supernet-tiny.pth
```

### With Fine-tuning

Add `--finetune_epochs 40` to fine-tune the best architecture after proxy evaluation:

```bash
python random_sample_eval.py \
    --proxy MLFE \
    --space tiny \
    --num_samples 100 \
    --seed 42 \
    --output results_tiny_ft.pkl \
    --data_root /path/to/ImageNet1K \
    --checkpoint /path/to/supernet-tiny.pth \
    --finetune_epochs 40 \
    --finetune_lr 5e-4 \
    --finetune_batch_size 128
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--proxy` | Proxy name (e.g. `MLFE`, `synflow`, `zen`) |
| `--space` | Search space: `tiny`, `small`, `base` |
| `--num_samples` | Number of random architectures to sample |
| `--seed` | Random seed |
| `--output` | Output pickle file path |
| `--data_root` | ImageNet dataset root |
| `--batch_size` | Batch size for proxy evaluation |
| `--checkpoint` | Path to pre-trained supernet checkpoint |
| `--finetune_epochs` | Fine-tune epochs (0 = no fine-tuning) |
| `--finetune_lr` | Fine-tune learning rate |
| `--finetune_batch_size` | Fine-tune batch size |

### Pipeline

1. **Proxy evaluation**: Sample N architectures → compute proxy score for each (using supernet weights)
2. **Subnet validation**: Evaluate the best architecture on ImageNet val set (no training, supernet weights only)
3. **Fine-tuning** (optional): Train the best architecture for K epochs on ImageNet, report final accuracy

## Files

```
autoformer/
├── random_sample_eval.py    # Main script: proxy eval + validate + finetune
├── model/
│   ├── autoformer_space.py  # Vision_TransformerSuper (supernet)
│   ├── autoformer_subnet.py # Subnet definitions
│   └── supernet_transformer.py
└── README.md
```

## AutoFormer Search Space

| Dimension | Tiny | Small | Base |
|-----------|------|-------|------|
| Embed dim | 192, 216, 240 | 320, 384, 448 | 528, 576, 624 |
| Depth | 12, 13, 14 | 12, 13, 14 | 14, 15, 16 |
| Num heads | 3, 4 | 5, 6, 7 | 9, 10 |
| MLP ratio | 3.5, 4.0 | 3.0, 3.5, 4.0 | 3.0, 3.5, 4.0 |

Sampled config format:
```python
{
    'embed_dim': [dim] * depth,
    'mlp_ratio': [ratio_per_layer],
    'num_heads': [heads_per_layer],
    'layer_num': depth
}
```

## Supported Proxies

| Proxy | Description |
|-------|-------------|
| MLFE | Mixed-Granularity Local Feature Entropy |
| ES | Entropy Score |
| synflow | Synaptic Flow |
| zen | Zen score |
| zico | ZICO proxy |
| meco | MECO proxy |
| naswot | NASWOT score |
| swap | SWAP proxy |
| near | NEAR proxy |
| epads | EPADS proxy |
| wrcor | Weighted Correlation |
| dextr | DEXTR proxy |
| epsinas | EP-NAS proxy |
| fisher | Fisher information |
| snip | SNIP proxy |
| grasp | GraSP proxy |
| jacob_cov | Jacobian covariance |
| grad_norm | Gradient norm |

## MLFE Adaptation for Vision Transformers

For applying MLFE to Vision Transformers, the output token sequence of each transformer block
is taken as the feature map. Given a feature map of shape `Batch × T × D` (where `T` is the
sequence length and `D` is the model dimension), the local entropy is computed over a
single-token embedding of dimension `D`. Tokens are uniformly sampled from the sequence for
entropy computation. All other settings remain consistent with the default MLFE configuration.

## Output

By default, results are saved to `../results/autoformer/result.pkl` as pickle files containing:
- `best_config`: Best architecture config
- `best_proxy_score`: Best proxy score
- `best_params`: Parameter count of best architecture
- `all_results`: All sampled architectures with proxy scores
- `subnet_val_acc1` / `subnet_val_acc5`: Subnet validation accuracy (if checkpoint provided)
- `finetune_acc1` / `finetune_acc5`: Fine-tune accuracy (if `--finetune_epochs > 0`)

## Acknowledgments

This project builds upon the following open-source works:

- **AutoFormer / Cream** (Microsoft) — AutoFormer supernet and Vision Transformer search space

We gratefully acknowledge these projects for providing the foundation on which this work is built.
