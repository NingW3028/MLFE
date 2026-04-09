"""
NAS-Bench-101 Zero-Cost Proxy Evaluation
Evaluates zero-cost proxies across architectures in NAS-Bench-101.
Computes proxy values, Spearman correlation, and Top-K accuracy.

Supports two data loading modes:
  --mode file : Load pre-computed architecture .npy files (from arc_select)
  --mode api  : Query NAS-Bench-101 API directly (requires TF via tf conda env)

Dataset: CIFAR-10
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# If TensorFlow is in a separate environment, set TF_SITE_PACKAGES to that env's site-packages path.
# Example: export TF_SITE_PACKAGES=/path/to/tf_env/lib/python3.7/site-packages
if os.environ.get('TF_SITE_PACKAGES'):
    sys.path.append(os.environ['TF_SITE_PACKAGES'])
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
import torchvision.transforms as transforms
import torchvision.datasets

from nasbench_pytorch.model import Network as NBNetwork

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
import pruners.predictive


# ============================================================
# Utility: clear invalid edges in adjacency matrix
# ============================================================

def clear_matrix(matrix):
    matrix_shape = np.shape(matrix)
    V = 1
    for i in matrix_shape:
        V = i * V
    H = int(np.sqrt(V))
    matrix = np.reshape(matrix, (H, H))
    while True:
        pass_value = True
        for i in range(H):
            if i == H - 1:
                continue
            elif i != H - 1 and i > 0:
                if np.sum(matrix[i, :]) == 0 and np.sum(matrix[:, i]) > 0:
                    matrix[:, i] = 0
                    pass_value = False
            elif i == 0:
                if np.sum(matrix[i, :]) == 0:
                    matrix = matrix * 0
                    return np.reshape(matrix, matrix_shape)
        Connect_beg_Stat = [-1 for i in range(H)]
        Connect_beg_Stat[0] = 1
        connect_beg_list = list(np.where(matrix[0, :] == 1)[0])
        while len(connect_beg_list) > 0:
            index = connect_beg_list[0]
            Connect_beg_Stat[index] = 1
            connect_beg_list.extend(list(np.where(matrix[index, :] == 1)[0]))
            del connect_beg_list[0]
        for i in range(H):
            if Connect_beg_Stat[i] == -1:
                matrix[:, i] = 0
                matrix[i, :] = 0
        Connect_end_Stat = [-1 for i in range(H)]
        Connect_end_Stat[-1] = 1
        Connect_end_list = list(np.where(matrix[:, -1] == 1)[0])
        while len(Connect_end_list) > 0:
            index = Connect_end_list[0]
            Connect_end_Stat[index] = 1
            Connect_end_list.extend(list(np.where(matrix[:, index] == 1)[0]))
            del Connect_end_list[0]
        for i in range(H):
            if Connect_end_Stat[i] == -1:
                matrix[:, i] = 0
                matrix[i, :] = 0
        if pass_value:
            return np.reshape(matrix, matrix_shape)


# ============================================================
# Data Loader
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
    elif proxy_name in ['snip', 'grasp', 'fisher', 'jacob_cov', 'grad_norm', 'ntk']:
        # Use pruners.predictive for these proxies (same as original TENAS)
        proj_queue = cifar10_dataloaders(64)
        metric = pruners.predictive.find_measures(network, proj_queue,
                    ('random', 1, 10), torch.device("cuda"), measure_names=[proxy_name])
        return metric[proxy_name], 0.0
    else:
        raise ValueError(f"Unknown proxy: {proxy_name}")


# ============================================================
# Main Evaluation
# ============================================================

# ============================================================
# Data Loading: File Mode (from pre-computed .npy files)
# ============================================================

def load_architectures_from_file(arc_data_path, num_archs=None):
    """Load pre-computed architecture data from .npy files."""
    print(f'Loading architecture data from {arc_data_path}...', flush=True)
    keep_indi_all = np.load(os.path.join(arc_data_path, 'keep_indi_all.npy'), allow_pickle=True)
    keep_arc_all = np.load(os.path.join(arc_data_path, 'keep_arc_all.npy'), allow_pickle=True)
    keep_matrix_list_all = np.load(os.path.join(arc_data_path, 'keep_matrix_list_all.npy'), allow_pickle=True)

    api_len = len(keep_matrix_list_all)
    if num_archs and num_archs < api_len:
        indices = sorted(random.sample(range(api_len), num_archs))
        keep_indi_all = keep_indi_all[indices]
        keep_arc_all = keep_arc_all[indices]
        keep_matrix_list_all = keep_matrix_list_all[indices]
        api_len = num_archs

    print(f'Loaded {api_len} architectures (file mode)', flush=True)
    return keep_indi_all, keep_arc_all, keep_matrix_list_all, api_len


# ============================================================
# Data Loading: API Mode (query NAS-Bench-101 directly)
# ============================================================

def load_architectures_from_api(api_path, num_archs=None):
    """Load architectures directly from NAS-Bench-101 API (requires TF)."""
    print(f'Loading NAS-Bench-101 API from {api_path}...', flush=True)
    # TF 2.x compatibility: nasbench uses tf.python_io (TF 1.x API)
    import tensorflow as tf
    if not hasattr(tf, 'python_io'):
        tf.python_io = tf.compat.v1.python_io
    if not hasattr(tf, 'gfile'):
        tf.gfile = tf.compat.v1.gfile
    from nasbench import api as nb_api
    nasbench = nb_api.NASBench(api_path)

    all_hashes = list(nasbench.hash_iterator())
    total = len(all_hashes)
    print(f'NAS-Bench-101 API loaded: {total} unique architectures', flush=True)

    if num_archs and num_archs < total:
        selected_hashes = random.sample(all_hashes, num_archs)
    else:
        selected_hashes = all_hashes

    keep_indi_all = []
    keep_arc_all = []
    keep_matrix_list_all = []

    for h in selected_hashes:
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(h)
        matrix = np.array(fixed_stats['module_adjacency'])
        ops = fixed_stats['module_operations']
        # Average test accuracy across all epochs=108 runs
        test_accs = [run['final_test_accuracy'] for run in computed_stats[108]]
        avg_acc = np.mean(test_accs)
        # Build indicator: [train_acc, valid_acc, test_acc, train_time, test_accuracy]
        # Index 4 = test_accuracy (same as file mode)
        indi = [0, 0, 0, 0, avg_acc]
        keep_indi_all.append(indi)
        keep_arc_all.append(ops)
        keep_matrix_list_all.append(matrix)

    api_len = len(keep_matrix_list_all)
    print(f'Loaded {api_len} architectures (api mode)', flush=True)
    return np.array(keep_indi_all), np.array(keep_arc_all, dtype=object), np.array(keep_matrix_list_all, dtype=object), api_len


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_proxy_on_benchmark(mode, arc_data_path, api_path, proxy_name, data_root,
                                batch_size=64, seed=42, output_dir='./results', num_archs=None):
    """Evaluate a single proxy across NAS-101 architectures."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load architecture data
    if mode == 'file':
        keep_indi_all, keep_arc_all, keep_matrix_list_all, api_len = \
            load_architectures_from_file(arc_data_path, num_archs)
    elif mode == 'api':
        keep_indi_all, keep_arc_all, keep_matrix_list_all, api_len = \
            load_architectures_from_api(api_path, num_archs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load data - same as reference
    train_loader = cifar10_dataloaders(batch_size, os.path.join(data_root, 'cifar10'))
    proj_queue = train_loader
    dataset = 'cifar10'

    os.makedirs(output_dir, exist_ok=True)

    proxy_values = []
    acc_values = []
    time_values = []
    time_begin = time.time()

    print(f'\n=== Evaluating {proxy_name} on NAS-101 ({api_len} architectures, mode={mode}) ===', flush=True)
    for arc_index in range(api_len):
        # Get data batch - same pattern as reference
        try:
            x, tar = next(proj_queue)
        except:
            proj_queue = iter(train_loader)
            x, tar = next(proj_queue)

        # Create network - same as reference
        raw_matrix = keep_matrix_list_all[arc_index]
        matrix = clear_matrix(raw_matrix.tolist() if hasattr(raw_matrix, 'tolist') else list(raw_matrix))
        raw_ops = keep_arc_all[arc_index]
        ops = raw_ops.tolist() if hasattr(raw_ops, 'tolist') else list(raw_ops)
        network = NBNetwork((matrix, ops)).cuda()

        # Get accuracy (index 4 = test_accuracy)
        acc = keep_indi_all[arc_index][4]
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
            _save_results(output_dir, proxy_name, seed, proxy_values, acc_values, time_values)

    # Final save & correlation
    _save_results(output_dir, proxy_name, seed, proxy_values, acc_values, time_values)

    proxy_arr = np.array(proxy_values, dtype=float)
    acc_arr = np.array(acc_values, dtype=float)
    proxy_arr[np.isnan(proxy_arr)] = 0
    proxy_arr[np.isinf(proxy_arr)] = 0

    corr = stats.spearmanr(proxy_arr, acc_arr)
    print(f'\n  {proxy_name} on cifar10: Spearman = {corr.correlation:.2f} (p={corr.pvalue:.2e})', flush=True)

    top10_idx = np.argsort(proxy_arr)[-10:]
    top10_acc = acc_arr[top10_idx]
    print(f'  Top-10 mean acc: {np.mean(top10_acc):.2f}', flush=True)

    top50_idx = np.argsort(proxy_arr)[-50:]
    top50_acc = acc_arr[top50_idx]
    print(f'  Top-50 mean acc: {np.mean(top50_acc):.2f}', flush=True)


def _save_results(output_dir, proxy_name, seed, proxy_values, acc_values, time_values):
    save_dir = os.path.join(output_dir, 'NAS101', 'cifar10', proxy_name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'proxy_seed_{seed}.npy'), proxy_values)
    np.save(os.path.join(save_dir, f'acc_seed_{seed}.npy'), acc_values)
    np.save(os.path.join(save_dir, f'time_seed_{seed}.npy'), time_values)


def run_multi_seed(mode, arc_data_path, api_path, proxy_name, data_root, batch_size, num_runs, base_seed, output_dir, num_archs=None):
    """Run evaluation across multiple seeds and aggregate results."""
    all_corrs = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f'\n{"="*60}', flush=True)
        print(f'Run {run_idx+1}/{num_runs}, seed={seed}', flush=True)
        print(f'{"="*60}', flush=True)

        evaluate_proxy_on_benchmark(mode, arc_data_path, api_path, proxy_name, data_root,
                                    batch_size, seed, output_dir, num_archs)

        save_dir = os.path.join(output_dir, 'NAS101', 'cifar10', proxy_name)
        proxy_arr = np.load(os.path.join(save_dir, f'proxy_seed_{seed}.npy'), allow_pickle=True).astype(float)
        acc_arr = np.load(os.path.join(save_dir, f'acc_seed_{seed}.npy'), allow_pickle=True).astype(float)
        proxy_arr[np.isnan(proxy_arr)] = 0
        proxy_arr[np.isinf(proxy_arr)] = 0
        corr = stats.spearmanr(proxy_arr, acc_arr)
        all_corrs.append(corr.correlation)

    corrs = np.array(all_corrs)
    print(f'\n{"="*60}', flush=True)
    print(f'SUMMARY: {proxy_name} over {num_runs} runs', flush=True)
    print(f'  cifar10: Spearman = {np.mean(corrs):.2f} +/- {np.std(corrs):.2f}', flush=True)
    print(f'{"="*60}', flush=True)

    summary_path = os.path.join(output_dir, 'NAS101', f'summary_{proxy_name}.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump({'proxy': proxy_name, 'num_runs': num_runs, 'base_seed': base_seed, 'correlations': all_corrs}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS-Bench-101 Zero-Cost Proxy Evaluation')
    parser.add_argument('--proxy', type=str, required=True,
                        help='Proxy to evaluate (MLFE, ES, synflow, meco, zico, zen, naswot, wrcor, '
                             'epads, near, dextr, epsinas, swap, snip, grasp, fisher, jacob_cov, grad_norm, ntk)')
    parser.add_argument('--mode', type=str, default='file', choices=['file', 'api'],
                        help='Data loading mode: file (pre-computed .npy) or api (NAS-Bench-101 API, requires TF)')
    parser.add_argument('--arc_data_path', type=str, default='./data',
                        help='Path to directory containing keep_indi_all.npy, keep_arc_all.npy, keep_matrix_list_all.npy')
    parser.add_argument('--api_path', type=str, default=None,
                        help='Path to nasbench_only108.tfrecord (api mode, required if --mode api)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for CIFAR-10 dataset (auto-downloaded if absent)')
    parser.add_argument('--num_archs', type=int, default=None,
                        help='Number of architectures to evaluate (random sample). None=all')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs with different seeds (1=single run)')
    parser.add_argument('--output', type=str, default='../results/ProxyEval-NAS101',
                        help='Output directory')
    args = parser.parse_args()

    if args.mode == 'api' and args.api_path is None:
        parser.error('--api_path is required when --mode api (path to nasbench_only108.tfrecord)')

    if args.num_runs > 1:
        run_multi_seed(args.mode, args.arc_data_path, args.api_path, args.proxy, args.data_root,
                       args.batch_size, args.num_runs, args.seed, args.output, args.num_archs)
    else:
        evaluate_proxy_on_benchmark(args.mode, args.arc_data_path, args.api_path, args.proxy, args.data_root,
                                    args.batch_size, args.seed, args.output, args.num_archs)
