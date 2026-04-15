"""
Evaluate MLFE proxy on TransNAS-Bench-101 (macro + micro search spaces).
Supports multiple input sources: Taskonomy task datasets.
"""
import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import re
from tqdm import tqdm
from threading import Thread
from queue import Queue
import torchvision.transforms as transforms
from PIL import Image

from api import TransNASBenchAPI
from transnas_models import MacroNet


# ============================================================
# Data loading
# ============================================================

# Taskonomy normalization params per task (from naslib configs)
TASKONOMY_NORM = {
    'class_scene':      {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]},
    'class_object':     {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]},
    'autoencoder':      {'mean': [0.5, 0.5, 0.5],          'std': [0.5, 0.5, 0.5]},
    'normal':           {'mean': [0.5, 0.5, 0.5],          'std': [0.5, 0.5, 0.5]},
    'jigsaw':           {'mean': [0.5, 0.5, 0.5],          'std': [0.5, 0.5, 0.5]},
    'room_layout':      {'mean': [0.5, 0.5, 0.5],          'std': [0.5, 0.5, 0.5]},
    'segmentsemantic':  {'mean': [0.5, 0.5, 0.5],          'std': [0.5, 0.5, 0.5]},
}

TASKONOMY_TASKS = list(TASKONOMY_NORM.keys())


def _find_taskonomy_images(data_dir):
    """Search for Taskonomy RGB images in common directory layouts."""
    patterns = [
        os.path.join(data_dir, '**', 'rgb', '*.png'),
        os.path.join(data_dir, '**', '*rgb*.png'),
        os.path.join(data_dir, '**', '*.png'),
    ]
    for pat in patterns:
        paths = sorted(glob.glob(pat, recursive=True))
        if paths:
            return paths
    raise FileNotFoundError(
        f"No RGB images found in {data_dir}. "
        f"Expected Taskonomy data at: {data_dir}/*/rgb/*.png")


def load_input_data(device, data_dir='./data', batch_size=1, input_source='ones'):
    """
    Load input tensor for forward pass.
    Args:
        input_source: 'ones' or a Taskonomy task name (e.g. 'class_scene')
        data_dir: root directory for dataset
        batch_size: number of images to load
    """
    if input_source == 'ones':
        return torch.ones(batch_size, 3, 64, 64).to(device)

    elif input_source in TASKONOMY_TASKS:
        # Load Taskonomy RGB images with task-specific normalization
        norm = TASKONOMY_NORM[input_source]
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(norm['mean'], norm['std']),
        ])
        image_paths = _find_taskonomy_images(data_dir)
        np.random.shuffle(image_paths)
        images = []
        for p in image_paths[:batch_size]:
            img = Image.open(p).convert('RGB')
            images.append(transform(img))
        x = torch.stack(images)
        print(f"  Loaded {len(images)} images for task '{input_source}' from {data_dir}")
        return x.to(device)

    else:
        raise ValueError(
            f"Unknown input_source: '{input_source}'. ")


# ============================================================
# Core computation functions
# ============================================================

def _get_patch_positions(h, w, conv_size, sqrt_cell_num):
    half = conv_size // 2
    i_list = [i + half for i in range(h - 2 * half - conv_size + 1)]
    j_list = [j + half for j in range(w - 2 * half - conv_size + 1)]
    if not i_list or not j_list:
        i_list = list(range(h - conv_size + 1))
        j_list = list(range(w - conv_size + 1))
    if sqrt_cell_num != 0 and len(i_list) > 1 and len(j_list) > 1:
        i_idx = np.linspace(0, len(i_list) - 1, sqrt_cell_num, endpoint=True).astype(int)
        j_idx = np.linspace(0, len(j_list) - 1, sqrt_cell_num, endpoint=True).astype(int)
        i_list = [i_list[i] for i in i_idx]
        j_list = [j_list[j] for j in j_idx]
    return i_list, j_list


def _entropy_products_for_matrix(matrix):
    """Vectorized discrete_entropy * continue_entropy on CPU."""
    batch_size, N = matrix.shape
    B = N
    min_val = matrix.min(dim=1, keepdim=True).values
    max_val = matrix.max(dim=1, keepdim=True).values
    val_range = (max_val - min_val).clamp(min=1e-10)
    constant_mask = ((max_val - min_val).squeeze(1) < 1e-10)
    bin_indices = ((matrix - min_val) / val_range * (B - 1)).long().clamp(0, B - 1)
    counts = torch.zeros(batch_size, B)
    counts.scatter_add_(1, bin_indices, torch.ones_like(matrix))
    mask = counts > 0
    probs_d = counts / counts.sum(dim=1, keepdim=True)
    log_p = torch.zeros_like(probs_d)
    log_p[mask] = torch.log2(probs_d[mask])
    d_entropy = -(probs_d * log_p).sum(dim=1)
    d_entropy[constant_mask] = 0.0
    probs_c = torch.softmax(torch.abs(matrix), dim=1)
    c_entropy = -torch.sum(probs_c * torch.log2(probs_c), dim=1)
    return d_entropy * c_entropy


def _extract_patches_per_convsize(fea, conv_size):
    h, w = fea.shape[1], fea.shape[2]
    if h < conv_size or w < conv_size:
        return None
    i_list, j_list = _get_patch_positions(h, w, conv_size, 4)
    patches = [fea[:, i:i + conv_size, j:j + conv_size].reshape(1, -1)
               for i in i_list for j in j_list]
    if not patches:
        return None
    return torch.cat(patches, dim=0)


def _aggregate_score(layer_sums):
    results = torch.tensor(layer_sums)
    results[torch.isnan(results)] = 0
    results[torch.isinf(results)] = 0
    non_zero_mask = results != 0
    non_zero_elements = results[non_zero_mask]
    non_zero_elements, _ = torch.sort(non_zero_elements)
    if len(non_zero_elements) > 0:
        len_nz = len(non_zero_elements)
        diff_list = []
        for i in range(len_nz):
            diff = (non_zero_elements[(len_nz - 1) - i] - non_zero_elements[i]).item()
            if (len_nz - 1 - i) < i:
                break
            diff_list.append(diff)
        if len(diff_list) > 0:
            diff_t = torch.tensor(diff_list)
            v = max(torch.max(diff_t).item(), 1e-10) * torch.sum(results).item()
        else:
            v = torch.sum(results).item()
    else:
        v = 0.0
    return v


def _batch_compute_scores_cpu(nets_fea_maps_cpu):
    n_archs = len(nets_fea_maps_cpu)
    max_layers = max(len(fm_list) for fm_list in nets_fea_maps_cpu)
    arch_layer_sums = [[0.0] * max_layers for _ in range(n_archs)]

    for conv_size in range(1, 16):
        dim_groups = {}
        for arch_idx, fm_list in enumerate(nets_fea_maps_cpu):
            for layer_idx, fea in enumerate(fm_list):
                mat = _extract_patches_per_convsize(fea, conv_size)
                if mat is not None:
                    d = mat.shape[1]
                    if d not in dim_groups:
                        dim_groups[d] = ([], [])
                    dim_groups[d][0].append(mat)
                    dim_groups[d][1].extend([(arch_idx, layer_idx)] * mat.shape[0])

        for d, (mat_list, idx_list) in dim_groups.items():
            big_matrix = torch.cat(mat_list, dim=0)
            products = _entropy_products_for_matrix(big_matrix)
            products[torch.isnan(products)] = 0
            products[torch.isinf(products)] = 0
            offset = 0
            for mat in mat_list:
                n = mat.shape[0]
                chunk = products[offset:offset + n]
                non_zero = chunk[chunk != 0]
                s = non_zero.sum().item() if len(non_zero) > 0 else 0.0
                ai, li = idx_list[offset]
                arch_layer_sums[ai][li] += s
                offset += n

    scores = []
    for arch_idx in range(n_archs):
        scores.append(_aggregate_score(arch_layer_sums[arch_idx]))
    return scores


# ============================================================
# Feature map collection (GPU, with configurable hook pattern)
# ============================================================


def _collect_feature_maps(net, x, device, dtype, layer_regex):
    """Forward pass on GPU, collect feature maps and move to CPU."""
    fea_maps = []

    def forward_hook(module, data_input, data_output):
        if isinstance(data_output, torch.Tensor):
            fea = data_output[0].detach()
        elif isinstance(data_output, tuple) and len(data_output) > 0:
            fea = data_output[0][0].detach() if isinstance(data_output[0], torch.Tensor) else data_output[0].detach()
        else:
            return
        fea[torch.isnan(fea)] = 0
        fea[torch.isinf(fea)] = 0
        if len(fea.shape) >= 2:
            fea_maps.append(fea.cpu())

    hooks = []
    for name, module in net.named_modules():
        if re.match(layer_regex, name):
            hooks.append(module.register_forward_hook(forward_hook))

    net.zero_grad()
    with torch.no_grad():
        net(x)

    for h in hooks:
        h.remove()
    return fea_maps


# ============================================================
# GPU producer thread
# ============================================================

def _gpu_producer(arch_list, eval_order, x, device, dtype, layer_regex,
                  fea_queue, N):
    batch_fea = []
    batch_indices = []
    failed = 0

    for step in range(N):
        i = eval_order[step]
        arch = arch_list[i]
        try:
            net = MacroNet(arch, structure='full', input_dim=(64, 64), num_classes=75)
            net = net.to(device).to(dtype)
            net.eval()
            fea_maps = _collect_feature_maps(net, x, device, dtype, layer_regex)
            batch_fea.append(fea_maps)
            batch_indices.append(i)
            del net
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Warning: {arch} failed: {e}", flush=True)
            fea_queue.put(('fail', i, step))
            continue

        if len(batch_fea) >= 1 or step == N - 1:
            fea_queue.put(('batch', list(batch_indices), batch_fea[:]))
            batch_fea.clear()
            batch_indices.clear()
            if device == 'cuda':
                torch.cuda.empty_cache()

    fea_queue.put(('done', None, None))


# ============================================================
# Ground truth
# ============================================================

def get_ground_truth_metric(api, arch, task):
    metrics = api.metrics_dict[task]
    if 'test_top1' in metrics:
        return api.get_single_metric(arch, task, 'test_top1', mode='best')
    elif 'test_ssim' in metrics:
        return api.get_single_metric(arch, task, 'test_ssim', mode='best')
    elif 'test_mIoU' in metrics:
        return api.get_single_metric(arch, task, 'test_mIoU', mode='best')
    elif 'test_neg_loss' in metrics:
        return api.get_single_metric(arch, task, 'test_neg_loss', mode='best')
    else:
        return None


# ============================================================
# Evaluate one search space
# ============================================================

def evaluate_search_space(api, space_name, device, dtype, x=None, pipeline=False, task=None):
    if task is not None:
        task_list = [task]
    else:
        task_list = api.task_list
    arch_list = list(api.all_arch_dict[space_name])
    N = len(arch_list)
    # For macro space, architectures differ in topology across all layers, so hook all.
    # For micro space, architectures share the same macro skeleton and only differ in
    # cell structure, so a single representative layer suffices.
    if space_name == 'macro':
        layer_regex = r'^layer\d+'
    else:
        layer_regex = r'^layer1'

    eval_order = list(range(N))
    np.random.shuffle(eval_order)

    proxy_scores = [0.0] * N
    acc_dict = {task: [] for task in task_list}

    for arch in arch_list:
        for task in task_list:
            gt = get_ground_truth_metric(api, arch, task)
            acc_dict[task].append(gt if gt is not None else 0.0)

    if x is None:
        x = torch.ones(1, 3, 64, 64).to(device)

    t0 = time.time()
    failed = 0

    if pipeline:
        # Pipeline mode: GPU forward in producer thread, CPU entropy in main thread
        evaluated_count = 0
        fea_queue = Queue(maxsize=2)

        producer = Thread(
            target=_gpu_producer,
            args=(arch_list, eval_order, x, device, dtype, layer_regex, fea_queue, N),
            daemon=True
        )
        producer.start()

        pbar = tqdm(total=N, desc=f'[{space_name}]', ncols=100)
        while True:
            msg = fea_queue.get()
            msg_type = msg[0]

            if msg_type == 'done':
                break
            elif msg_type == 'fail':
                failed += 1
                evaluated_count += 1
                pbar.update(1)
                pbar.set_postfix(failed=failed)
            elif msg_type == 'batch':
                indices = msg[1]
                batch_fea = msg[2]

                scores = _batch_compute_scores_cpu(batch_fea)

                for idx, score in zip(indices, scores):
                    proxy_scores[idx] = score

                evaluated_count += len(indices)
                pbar.update(len(indices))
                pbar.set_postfix(failed=failed)

                if evaluated_count % 50 < 16 and evaluated_count >= 50:
                    evaluated = eval_order[:evaluated_count]
                    tmp_proxy = np.array([proxy_scores[j] for j in evaluated], dtype=float)
                    tmp_proxy[np.isnan(tmp_proxy)] = 0
                    tmp_proxy[np.isinf(tmp_proxy)] = 0
                    corr_strs = []
                    for task in task_list:
                        tmp_acc = np.array([acc_dict[task][j] for j in evaluated], dtype=float)
                        tmp_acc[np.isnan(tmp_acc)] = 0
                        rho = stats.spearmanr(tmp_proxy, tmp_acc).correlation
                        corr_strs.append(f"{task[:8]}={rho:.3f}")
                    tqdm.write(f"    Spearman @{evaluated_count}: {', '.join(corr_strs)}")

                del batch_fea

        pbar.close()
        producer.join()
    else:
        # Sequential mode: forward + entropy in same thread
        batch_archs = []
        batch_indices = []
        pbar = tqdm(range(N), desc=f'[{space_name}]', ncols=100)
        for step in pbar:
            i = eval_order[step]
            arch = arch_list[i]
            try:
                net = MacroNet(arch, structure='full', input_dim=(64, 64), num_classes=75)
                net = net.to(device).to(dtype)
                net.eval()
                fea_maps = _collect_feature_maps(net, x, device, dtype, layer_regex)
                batch_archs.append([f.cpu() if f.is_cuda else f for f in fea_maps])
                batch_indices.append(i)
                del net
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  Warning: {arch} failed: {e}", flush=True)
                continue

            if len(batch_archs) >= 16 or step == N - 1:
                scores = _batch_compute_scores_cpu(batch_archs)
                for idx, score in zip(batch_indices, scores):
                    proxy_scores[idx] = score
                batch_archs.clear()
                batch_indices.clear()
                if device == 'cuda':
                    torch.cuda.empty_cache()

            pbar.set_postfix(failed=failed)
        pbar.close()

    elapsed = time.time() - t0
    print(f"  [{space_name}] Done: {N} archs in {elapsed:.2f}s (failed={failed})", flush=True)

    proxy_arr = np.array(proxy_scores, dtype=float)
    proxy_arr[np.isnan(proxy_arr)] = 0
    proxy_arr[np.isinf(proxy_arr)] = 0

    results = {}
    for task in task_list:
        acc_arr = np.array(acc_dict[task], dtype=float)
        acc_arr[np.isnan(acc_arr)] = 0
        corr = stats.spearmanr(proxy_arr, acc_arr)
        results[task] = (corr.correlation, corr.pvalue)

    return results, proxy_scores


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_path', type=str, required=True,
                        help='Path to TransNAS-Bench-101 .pth file')
    parser.add_argument('--eval_task', type=str, default='class_scene',
                        choices=TASKONOMY_TASKS,
                        help='Taskonomy task to evaluate correlation for')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Taskonomy data directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images to use as input')
    parser.add_argument('--pipeline', action='store_true',
                        help='Enable pipeline mode (GPU/CPU overlap)')
    parser.add_argument('--space', type=str, default='macro',
                        choices=['macro', 'micro'],
                        help='Search space to evaluate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    np.random.seed(42)

    x = load_input_data(device, args.data_dir, args.batch_size)

    api = TransNASBenchAPI(args.api_path)

    results, _ = evaluate_search_space(
        api, args.space, device, dtype, x=x, pipeline=args.pipeline, task=args.eval_task)
    print(f"\n{args.space.upper()} Spearman Correlations (eval_task={args.eval_task}):")
    for task, (rho, pval) in results.items():
        print(f"  {task:25s}  rho={rho:.4f}")

if __name__ == '__main__':
    main()
