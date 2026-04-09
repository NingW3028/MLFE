# ProxyEval-Trans101

Evaluate MLFE proxy on **TransNAS-Bench-101** (macro / micro search spaces) with Taskonomy dataset support.

## Files

| File | Description |
|---|---|
| `eval_proxy.py` | Main evaluation script |
| `api.py` | TransNASBenchAPI (load benchmark data) |
| `transnas_models.py` | MacroNet / MicroCell network builder |

## Requirements

- Python 3.7+
- PyTorch, torchvision, scipy, tqdm, numpy, Pillow

## Data

- **API file**: TransNAS-Bench-101 `.pth` file 
- **Taskonomy images** (optional): RGB image directory for Taskonomy tasks

## Usage

```bash
# Default: class_scene task, macro space, sequential mode
python eval_proxy.py --api_path /path/to/transnas-bench_v10141024.pth --data_dir /path/to/taskonomy_images
```

## Supported Tasks (Taskonomy)

`class_scene`, `class_object`, `autoencoder`, `normal`, `jigsaw`, `room_layout`, `segmentsemantic`
