"""
Random Sample Evaluation for AutoFormer Search Space (E4 Experiment)
Samples 1000 random architectures from the supernet, evaluates each with a proxy,
and selects the one with the highest proxy score.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
from tqdm import tqdm
from timm.utils.model import unwrap_model
from timm.loss import LabelSmoothingCrossEntropy

from model.autoformer_space import Vision_TransformerSuper

AUTOFORMER_SPACES = {
    'tiny': {
        'embed_dim': [192, 216, 240],
        'num_heads': [3, 4],
        'mlp_ratio': [3.5, 4.0],
        'depth': [12, 13, 14],
        'super_embed_dim': 256,
        'super_depth': 14,
        'super_num_heads': 4,
        'super_mlp_ratio': 4.0,
    },
    'small': {
        'embed_dim': [320, 384, 448],
        'num_heads': [5, 6, 7],
        'mlp_ratio': [3.0, 3.5, 4.0],
        'depth': [12, 13, 14],
        'super_embed_dim': 448,
        'super_depth': 14,
        'super_num_heads': 7,
        'super_mlp_ratio': 4.0,
    },
    'base': {
        'embed_dim': [528, 576, 624],
        'num_heads': [9, 10],
        'mlp_ratio': [3.0, 3.5, 4.0],
        'depth': [14, 15, 16],
        'super_embed_dim': 640,
        'super_depth': 16,
        'super_num_heads': 10,
        'super_mlp_ratio': 4.0,
    }
}


def sample_random_config(space):
    """Sample a random architecture config from the search space"""
    depth = random.choice(space['depth'])
    embed_dim = random.choice(space['embed_dim'])
    # Only choose num_heads that divides embed_dim
    valid_heads = [h for h in space['num_heads'] if embed_dim % h == 0]
    if not valid_heads:
        valid_heads = space['num_heads']  # fallback
    config = {
        'embed_dim': [embed_dim] * depth,
        'mlp_ratio': [random.choice(space['mlp_ratio']) for _ in range(depth)],
        'num_heads': [random.choice(valid_heads) for _ in range(depth)],
        'layer_num': depth,
    }
    return config


def create_supernet(space, device, checkpoint=None):
    """Create the supernet model, optionally load pre-trained weights"""
    model = Vision_TransformerSuper(
        img_size=224,
        patch_size=16,
        embed_dim=space['super_embed_dim'],
        depth=space['super_depth'],
        num_heads=space['super_num_heads'],
        mlp_ratio=space['super_mlp_ratio'],
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        gp=True,
        num_classes=1000,
        max_relative_position=14,
        relative_position=True,
        change_qkv=True,
        abs_pos=True
    )
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    if device == 'cuda':
        model = model.cuda()
    return model


def compute_proxy(net, inputs, targets, dtype, proxy_name):
    """Compute proxy score"""
    net.zero_grad()

    if proxy_name == 'synflow':
        from proxieslib.synflow import compute_synflow
        net(inputs)
        score, _ = compute_synflow(net, inputs, dtype, targets, mode='param')
    elif proxy_name == 'zen':
        from proxieslib.zen import compute_zen
        score, _ = compute_zen(net, inputs, targets, dtype)
    elif proxy_name == 'zico':
        from proxieslib.zico import compute_zico
        score, _ = compute_zico(net, inputs, targets)
    elif proxy_name == 'meco':
        from proxieslib.meco import compute_meco
        score, _ = compute_meco(net, inputs, targets)
    elif proxy_name == 'swap':
        from proxieslib.swap import compute_swap
        score = compute_swap(net, inputs, targets)
    elif proxy_name == 'naswot':
        from proxieslib.naswot import compute_naswot
        score = compute_naswot(net, inputs, targets)
    elif proxy_name == 'wrcor':
        from proxieslib.wrcor import compute_wrcor
        score = compute_wrcor(net, inputs, targets)
    elif proxy_name == 'epads':
        from proxieslib.epads import compute_epads
        score = compute_epads(net, inputs, targets)
    elif proxy_name == 'near':
        from proxieslib.near import compute_near
        score = compute_near(net, inputs, targets)
    elif proxy_name == 'dextr':
        from proxieslib.dextr import compute_dextr
        score = compute_dextr(net, inputs, targets)
    elif proxy_name == 'epsinas':
        from proxieslib.epsinas import compute_epsinas
        score = compute_epsinas(net, inputs, targets)
    elif proxy_name == 'MLFE':
        from proxieslib.MLFE import compute_MLFE
        score = compute_MLFE(net, inputs, dtype, benchtype='AutoFormer', dataset='ImageNet1K')
    elif proxy_name == 'ES':
        from proxieslib.EntropyScore import compute_EntropyScore
        score = compute_EntropyScore(net, inputs, dtype, benchtype='AutoFormer', dataset='ImageNet1K')
    elif proxy_name == 'fisher':
        from proxieslib.fisher import compute_fisher_per_weight
        score = compute_fisher_per_weight(net, inputs, targets, loss_fn=nn.CrossEntropyLoss())
    elif proxy_name == 'snip':
        from proxieslib.snip import compute_snip_per_weight
        score = compute_snip_per_weight(net, inputs, targets, loss_fn=nn.CrossEntropyLoss())
    elif proxy_name == 'grasp':
        from proxieslib.grasp import compute_grasp_per_weight
        score = compute_grasp_per_weight(net, inputs, targets, loss_fn=nn.CrossEntropyLoss())
    elif proxy_name == 'jacob_cov':
        from proxieslib.jacob_cov import compute_jacob_cov
        score = compute_jacob_cov(net, inputs, targets, loss_fn=nn.CrossEntropyLoss())
    elif proxy_name == 'grad_norm':
        from proxieslib.grad_norm import get_grad_norm_arr
        score = get_grad_norm_arr(net, inputs, targets, loss_fn=nn.CrossEntropyLoss())
    else:
        raise ValueError(f"Unknown proxy: {proxy_name}")

    return float(score) if score is not None else 0.0


def get_imagenet_loader(data_root, batch_size=16):
    """Get ImageNet train loader (small batch for proxy eval)"""
    traindir = os.path.join(data_root, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True)
    return train_loader


def get_imagenet_train_loader_finetune(data_root, batch_size=128):
    """Get ImageNet train loader for fine-tuning"""
    traindir = os.path.join(data_root, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True)
    return train_loader


def get_imagenet_val_loader(data_root, batch_size=256):
    """Get ImageNet validation loader"""
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_data = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=4)
    return val_loader


def finetune_and_eval(space, best_config, data_root, device,
                      epochs=40, lr=5e-4, batch_size=128, smoothing=0.1,
                      warmup_epochs=5, checkpoint=None):
    """Fine-tune the best architecture on ImageNet and evaluate.
    Follows AutoFormer training protocol:
    - AdamW optimizer
    - Cosine LR scheduler with warmup
    - Label smoothing cross-entropy
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning best architecture for {epochs} epochs")
    print(f"Config: depth={best_config['layer_num']}, embed_dim={best_config['embed_dim'][0]}")
    print(f"LR={lr}, batch_size={batch_size}, smoothing={smoothing}")
    print(f"{'='*60}")

    # Create model and fix config (load pre-trained supernet weights)
    model = create_supernet(space, device, checkpoint)
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=best_config)
    params = model_module.get_sampled_params_numel(best_config)
    print(f"Sampled model parameters: {params}")

    # Data loaders
    train_loader = get_imagenet_train_loader_finetune(data_root, batch_size)
    val_loader = get_imagenet_val_loader(data_root, batch_size * 2)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing).to(device)
    scaler = torch.cuda.amp.GradScaler()

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_acc1 = 0.0
    for epoch in range(epochs):
        model.train()
        model_module.set_sample_config(config=best_config)
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.2f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Validate every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc1, acc5 = validate(model, model_module, best_config, val_loader, device)
            if acc1 > best_acc1:
                best_acc1 = acc1
            print(f'Epoch {epoch+1}/{epochs}: loss={avg_loss:.2f}, '
                  f'val_acc1={acc1:.2f}%, val_acc5={acc5:.2f}%, best_acc1={best_acc1:.2f}%')
        else:
            print(f'Epoch {epoch+1}/{epochs}: loss={avg_loss:.2f}, lr={optimizer.param_groups[0]["lr"]:.2e}')

    # Final validation
    acc1, acc5 = validate(model, model_module, best_config, val_loader, device)
    print(f"\nFinal: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
    return acc1, acc5


@torch.no_grad()
def validate(model, model_module, config, val_loader, device):
    """Evaluate on ImageNet validation set"""
    model.eval()
    model_module.set_sample_config(config=config)
    correct1 = 0
    correct5 = 0
    total = 0
    for images, targets in tqdm(val_loader, desc='Validating', leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
        _, pred5 = outputs.topk(5, 1, True, True)
        pred5 = pred5.t()
        correct = pred5.eq(targets.view(1, -1).expand_as(pred5))
        correct1 += correct[:1].reshape(-1).float().sum(0).item()
        correct5 += correct[:5].reshape(-1).float().sum(0).item()
        total += targets.size(0)
    acc1 = 100.0 * correct1 / total
    acc5 = 100.0 * correct5 / total
    return acc1, acc5


def main():
    parser = argparse.ArgumentParser(description='Random Sample Evaluation for AutoFormer (E4)')
    parser.add_argument('--proxy', type=str, required=True, help='Proxy name')
    parser.add_argument('--space', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='AutoFormer space type (default: tiny)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of random architectures to sample (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='../results/autoformer/result.pkl',
                        help='Output pickle file path')
    parser.add_argument('--data_root', type=str, default='./data/imagenet',
                        help='Root directory for ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for proxy evaluation (default: 16)')
    parser.add_argument('--finetune_epochs', type=int, default=0,
                        help='Number of fine-tune epochs for best arch (default: 0, set 40 for E4)')
    parser.add_argument('--finetune_lr', type=float, default=5e-4,
                        help='Fine-tune learning rate (default: 5e-4)')
    parser.add_argument('--finetune_batch_size', type=int, default=128,
                        help='Fine-tune batch size (default: 128)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pre-trained supernet checkpoint (.pth)')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    space = AUTOFORMER_SPACES[args.space]

    print(f"Device: {device}")
    print(f"Proxy: {args.proxy}")
    print(f"Space: {args.space}")
    print(f"Num samples: {args.num_samples}")
    print(f"Seed: {args.seed}")

    # Load data
    print(f"Loading ImageNet from {args.data_root}...")
    train_loader = get_imagenet_loader(args.data_root, args.batch_size)

    # Get a fixed batch for evaluation
    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    dtype = torch.float32

    # Sample random architectures
    print(f"Sampling {args.num_samples} random architectures...")
    configs = [sample_random_config(space) for _ in range(args.num_samples)]

    # Evaluate each architecture
    print("Evaluating architectures...")
    results = []
    supernet = create_supernet(space, device, args.checkpoint)

    for i, config in enumerate(tqdm(configs, desc=f'Eval [{args.proxy}]')):
        try:
            model_module = unwrap_model(supernet)
            model_module.set_sample_config(config=config)
            supernet.eval()

            score = compute_proxy(supernet, inputs, targets, dtype, args.proxy)
            params = model_module.get_sampled_params_numel(config)

            if np.isnan(score) or np.isinf(score):
                score = 0.0

            results.append({
                'config': config,
                'proxy_score': score,
                'params': params,
            })
        except Exception as e:
            print(f"  [WARNING] Config {i} failed: {e}")
            results.append({
                'config': config,
                'proxy_score': 0.0,
                'params': 0,
            })

        # Periodically clear cache
        if i % 100 == 0 and device == 'cuda':
            torch.cuda.empty_cache()

    del supernet
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Find best
    valid_results = [r for r in results if r['proxy_score'] != 0.0]
    if valid_results:
        best = max(valid_results, key=lambda x: x['proxy_score'])
    else:
        best = results[0] if results else None

    # Save
    output_data = {
        'best_config': best['config'] if best else None,
        'best_proxy_score': best['proxy_score'] if best else 0.0,
        'best_params': best['params'] if best else 0,
        'all_results': results,
        'proxy': args.proxy,
        'space': args.space,
        'seed': args.seed,
        'num_samples': args.num_samples,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nResults saved to {args.output}")
    if best:
        print(f"Best proxy score: {best['proxy_score']:.2e}")
        print(f"Best config: depth={best['config']['layer_num']}, "
              f"embed_dim={best['config']['embed_dim'][0]}")
        print(f"Best params: {best['params']}")
    print(f"Valid evaluations: {len(valid_results)}/{len(results)}")

    # Validate best subnet on ImageNet val (before fine-tune)
    if best and best['proxy_score'] != 0.0 and args.checkpoint:
        print(f"\nValidating best subnet (no fine-tune)...")
        val_loader = get_imagenet_val_loader(args.data_root, batch_size=256)
        val_model = create_supernet(space, device, args.checkpoint)
        val_module = unwrap_model(val_model)
        val_acc1, val_acc5 = validate(val_model, val_module, best['config'], val_loader, device)
        print(f"Subnet validation: Top-1={val_acc1:.2f}%, Top-5={val_acc5:.2f}%")
        output_data['subnet_val_acc1'] = val_acc1
        output_data['subnet_val_acc5'] = val_acc5
        del val_model, val_loader
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Fine-tune best architecture if requested
    if args.finetune_epochs > 0 and best and best['proxy_score'] != 0.0:
        acc1, acc5 = finetune_and_eval(
            space, best['config'], args.data_root, device,
            epochs=args.finetune_epochs, lr=args.finetune_lr,
            batch_size=args.finetune_batch_size,
            checkpoint=args.checkpoint)
        # Update output with fine-tune results
        output_data['finetune_acc1'] = acc1
        output_data['finetune_acc5'] = acc5
        output_data['finetune_epochs'] = args.finetune_epochs
        with open(args.output, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Updated results saved to {args.output}")


if __name__ == '__main__':
    main()
