"""
NAS-Bench-201 Zero-Cost Proxy Evaluation
Evaluates zero-cost proxies across all architectures in NAS-Bench-201.
Computes proxy values, Spearman correlation, and Top-K accuracy.

Datasets: CIFAR-10, CIFAR-100, ImageNet16-120
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time
import argparse
import pickle
import random
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from easydict import EasyDict as edict
import torchvision.transforms as transforms
import torchvision.datasets

from models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
from utilsfold.imagenet16 import ImageNet16
from torch.utils.data import DataLoader

from proxieslib.MLFE import compute_MLFE
from proxieslib.EntropyScore import compute_EntropyScore
from proxieslib.meco import compute_meco
from proxieslib.synflow import compute_synflow
from proxieslib.zico import compute_zico
from proxieslib.zen import compute_zen
from proxieslib.naswot import compute_naswot
from proxieslib.wrcorfast import compute_wrcorfast
from proxieslib.epads import compute_epads
from proxieslib.near import compute_near
from proxieslib.dextr import compute_dextr
from proxieslib.epsinas import compute_epsinas
from proxieslib.swap import SWAP


# ============================================================
# Data Loaders
# ============================================================

def cifar10_dataloaders(train_batch_size=64, data_dir='./data/cifar10'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return train_loader


def cifar100_dataloaders(train_batch_size=64, data_dir='./data/cifar100'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return train_loader


def imagenet16_dataloaders(train_batch_size=64, data_dir='./data/ImageNet16'):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std = [x / 255 for x in [63.22, 61.26, 65.09]]
    train_transform = transforms.Compose([
        transforms.RandomCrop(16, padding=2),
        transforms.Resize(16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = ImageNet16(data_dir, True, train_transform, 120)
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return train_loader


# ============================================================
# Proxy Computation
# ============================================================

def compute_proxy_score(network, x, tar, dtype, proxy_name, benchtype='tss', dataset='cifar10'):
    """Compute a single proxy score for a network."""
    if proxy_name == 'MLFE':
        beg = time.time()
        score = compute_MLFE(network, x, dtype, benchtype=benchtype, dataset=dataset)
        return score, time.time() - beg
    elif proxy_name == 'ES':
        beg = time.time()
        score = compute_EntropyScore(network, x, dtype, benchtype=benchtype, dataset=dataset)
        return score, time.time() - beg
    elif proxy_name == 'synflow':
        score, t = compute_synflow(network, x, tar, mode='param')
        return score, t
    elif proxy_name == 'meco':
        score, t = compute_meco(network, x, tar)
        return score, t
    elif proxy_name == 'zico':
        score, t = compute_zico(network, x, tar)
        return score, t
    elif proxy_name == 'zen':
        score, t = compute_zen(network, x, tar)
        return score, t
    elif proxy_name == 'naswot':
        beg = time.time()
        score = compute_naswot(network, x, dtype, benchtype=benchtype, dataset=dataset)
        return score, time.time() - beg
    elif proxy_name == 'wrcor':
        score, t = compute_wrcorfast(network, x, tar)
        return score, t
    elif proxy_name == 'epads':
        score, t = compute_epads(network, x, tar)
        return score, t
    elif proxy_name == 'near':
        score, t = compute_near(network, x, tar)
        return score, t
    elif proxy_name == 'dextr':
        score, t = compute_dextr(network, x, tar)
        return score, t
    elif proxy_name == 'epsinas':
        score, t = compute_epsinas(network, x, tar)
        return score, t
    elif proxy_name == 'swap':
        from proxieslib.swap import network_weight_gaussian_init
        beg = time.time()
        swap = SWAP(model=network, inputs=x, device='cuda', seed=0)
        swap_scores = []
        network_temp = network.apply(network_weight_gaussian_init)
        for _ in range(5):
            network_temp = network_temp.apply(network_weight_gaussian_init)
            swap.reinit()
            swap_scores.append(swap.forward())
            swap.clear()
        return np.mean(swap_scores), time.time() - beg
    else:
        raise ValueError(f"Unknown proxy: {proxy_name}")


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_proxy_on_benchmark(api, proxy_name, datasets, data_root, batch_size=64, seed=42, output_dir='./results'):
    """Evaluate a single proxy across all architectures and datasets."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    api_len = len(api)
    os.makedirs(output_dir, exist_ok=True)

    # Load data - same as reference 201tss_zero_cost_proxpy_v2.py
    loaders = {}
    if 'cifar10' in datasets:
        loaders['cifar10'] = cifar10_dataloaders(batch_size, os.path.join(data_root, 'cifar10'))
    if 'cifar100' in datasets:
        loaders['cifar100'] = cifar100_dataloaders(batch_size, os.path.join(data_root, 'cifar100'))
    if 'ImageNet16-120' in datasets:
        loaders['ImageNet16-120'] = imagenet16_dataloaders(batch_size, os.path.join(data_root, 'ImageNet16'))

    for dataset in datasets:
        print(f'\n=== Evaluating {proxy_name} on {dataset} ({api_len} architectures) ===', flush=True)
        train_loader = loaders[dataset]
        proj_queue = train_loader
        ImageNet16_data = True

        proxy_values = []
        acc_values = []
        time_values = []

        n_classes = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}[dataset]
        time_begin = time.time()

        for arc_index, arch_str in enumerate(api):
            # Get data batch - same pattern as reference
            if dataset == 'ImageNet16-120':
                if ImageNet16_data:
                    try:
                        x, tar = next(proj_queue)
                    except:
                        proj_queue = iter(train_loader)
                        x, tar = next(proj_queue)
                    ImageNet16_data = False
            else:
                try:
                    x, tar = next(proj_queue)
                except:
                    proj_queue = iter(train_loader)
                    x, tar = next(proj_queue)

            # Get architecture info - same as reference
            index = api.query_index_by_arch(arch_str)
            information = api.arch2infos_full[index]
            if dataset == 'cifar10-valid':
                continue
            config = edict(api.get_net_config(index, dataset))

            # Create network
            network = get_cell_based_tiny_net(config)
            network.cuda()

            # Get accuracy - same as reference
            if dataset == 'cifar10':
                test_info = information.get_metrics(dataset, 'ori-test')
            else:
                test_info = information.get_metrics(dataset, 'x-test')
            acc = test_info['accuracy']
            acc_values.append(acc)

            # Get dtype - same as reference
            for name, param in network.named_parameters():
                dtype = param.data.dtype
                continue
            x = torch.tensor(x, dtype=dtype)
            x = x.cuda()
            tar = tar.cuda()

            # Compute proxy
            try:
                score, elapsed = compute_proxy_score(network, x, tar, dtype, proxy_name, benchtype='tss', dataset=dataset)
                proxy_values.append(score)
                time_values.append(elapsed)
            except Exception as e:
                print(f'\n  Warning: arch {arc_index} failed: {e}', flush=True)
                proxy_values.append(0.0)
                time_values.append(0.0)

            del network
            torch.cuda.empty_cache()

            # Periodic report & save
            if (arc_index + 1) % 50 == 0:
                p = np.array(proxy_values, dtype=float)
                a = np.array(acc_values, dtype=float)
                p[np.isnan(p) | np.isinf(p)] = 0
                corr = stats.spearmanr(p, a)
                elapsed_total = time.time() - time_begin
                print(f'  [{arc_index+1:5d}/{api_len}] Spearman={corr.correlation:.2f}  time={elapsed_total:.2f}s', flush=True)
                _save_results(output_dir, dataset, proxy_name, seed, proxy_values, acc_values, time_values)

        # Final save & correlation
        _save_results(output_dir, dataset, proxy_name, seed, proxy_values, acc_values, time_values)

        proxy_arr = np.array(proxy_values, dtype=float)
        acc_arr = np.array(acc_values, dtype=float)
        proxy_arr[np.isnan(proxy_arr)] = 0
        proxy_arr[np.isinf(proxy_arr)] = 0

        corr = stats.spearmanr(proxy_arr, acc_arr)
        print(f'\n  {proxy_name} on {dataset}: Spearman = {corr.correlation:.2f} (p={corr.pvalue:.2e})', flush=True)

        top10_idx = np.argsort(proxy_arr)[-10:]
        top10_acc = acc_arr[top10_idx]
        print(f'  Top-10 mean acc: {np.mean(top10_acc):.2f}', flush=True)

        top50_idx = np.argsort(proxy_arr)[-50:]
        top50_acc = acc_arr[top50_idx]
        print(f'  Top-50 mean acc: {np.mean(top50_acc):.2f}', flush=True)


def _save_results(output_dir, dataset, proxy_name, seed, proxy_values, acc_values, time_values):
    """Save intermediate/final results."""
    save_dir = os.path.join(output_dir, 'NAS201', dataset, proxy_name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'proxy_seed_{seed}.npy'), proxy_values)
    np.save(os.path.join(save_dir, f'acc_seed_{seed}.npy'), acc_values)
    np.save(os.path.join(save_dir, f'time_seed_{seed}.npy'), time_values)


# ============================================================
# Multi-seed Runner
# ============================================================

def run_multi_seed(api, proxy_name, datasets, data_root, batch_size, num_runs, base_seed, output_dir):
    """Run evaluation across multiple seeds and aggregate results."""
    all_corrs = {ds: [] for ds in datasets}

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f'\n{"="*60}')
        print(f'Run {run_idx+1}/{num_runs}, seed={seed}')
        print(f'{"="*60}')

        evaluate_proxy_on_benchmark(api, proxy_name, datasets, data_root, batch_size, seed, output_dir)

        # Compute correlation for this run
        for dataset in datasets:
            save_dir = os.path.join(output_dir, 'NAS201', dataset, proxy_name)
            proxy_arr = np.load(os.path.join(save_dir, f'proxy_seed_{seed}.npy'), allow_pickle=True).astype(float)
            acc_arr = np.load(os.path.join(save_dir, f'acc_seed_{seed}.npy'), allow_pickle=True).astype(float)
            proxy_arr[np.isnan(proxy_arr)] = 0
            proxy_arr[np.isinf(proxy_arr)] = 0
            corr = stats.spearmanr(proxy_arr, acc_arr)
            all_corrs[dataset].append(corr.correlation)

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY: {proxy_name} over {num_runs} runs')
    print(f'{"="*60}')
    for dataset in datasets:
        corrs = np.array(all_corrs[dataset])
        print(f'  {dataset}: Spearman = {np.mean(corrs):.2f} +/- {np.std(corrs):.2f}')

    # Save summary
    summary_path = os.path.join(output_dir, 'NAS201', f'summary_{proxy_name}.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump({'proxy': proxy_name, 'num_runs': num_runs, 'base_seed': base_seed, 'correlations': all_corrs}, f)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS-Bench-201 Zero-Cost Proxy Evaluation')
    parser.add_argument('--proxy', type=str, required=True,
                        choices=['MLFE', 'ES', 'synflow', 'meco', 'zico', 'zen', 'naswot',
                                 'wrcor', 'epads', 'near', 'dextr', 'epsinas', 'swap'],
                        help='Proxy to evaluate')
    parser.add_argument('--api_path', type=str, required=True,
                        help='Path to NAS-Bench-201 API file (.pth), e.g. NAS-Bench-201-v1_0-e61699.pth')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['ImageNet16-120'],
                        help='Datasets to evaluate on (choices: cifar10, cifar100, ImageNet16-120)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets (all auto-downloaded; ImageNet16-120 requires gdown: pip install gdown)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs with different seeds (1=single run)')
    parser.add_argument('--output', type=str, default='../results/ProxyEval-NAS201',
                        help='Output directory')
    args = parser.parse_args()

    print(f'Loading NAS-Bench-201 API from {args.api_path}...', flush=True)
    t_api = time.time()
    api = API(args.api_path)
    print(f'Loaded {len(api)} architectures ({time.time()-t_api:.2f}s)', flush=True)

    if args.num_runs > 1:
        run_multi_seed(api, args.proxy, args.datasets, args.data_root,
                       args.batch_size, args.num_runs, args.seed, args.output)
    else:
        evaluate_proxy_on_benchmark(api, args.proxy, args.datasets, args.data_root,
                                    args.batch_size, args.seed, args.output)
