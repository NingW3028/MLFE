"""
Batch run all proxies on NAS-201, 30 runs per proxy
Uses separate processes to avoid memory accumulation
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import subprocess
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
PROXIES = [
    'epsinas', 'meco', 'zen', 'near', 'ES', 'synflow', 'naswot', 'MLFE',
    'epads', 'wrcor', 'zico', 'swap', 'dextr'
]

DATASET = 'cifar10'  # Change to: cifar10, cifar100, ImageNet16-120
NUM_RUNS = 30
MAX_EVALS = 500
BASE_SEED = 42

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'LocalSearch-nas201'
RESULTS_DIR.mkdir(exist_ok=True)

# Timestamp
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = RESULTS_DIR / f'run_{DATASET}_{TIMESTAMP}'
RUN_DIR.mkdir(exist_ok=True)


def run_single_experiment(run_id, proxy, seed):
    """Run single experiment using separate process"""
    output_file = RUN_DIR / f'run{run_id}__{proxy}__seed{seed}.pkl'

    # Check if already completed in current and historical directories
    filename = f'run{run_id}__{proxy}__seed{seed}.pkl'
    for run_dir in RESULTS_DIR.glob(f'run_{DATASET}_*'):
        existing_file = run_dir / filename
        if existing_file.exists():
            print(f"\n{'='*80}")
            print(f"Run {run_id+1}/{NUM_RUNS} | Proxy: {proxy} | Seed: {seed}")
            print(f"{'='*80}")
            print(f"[SKIP] Found in {run_dir.name}")
            return True

    print(f"\n{'='*80}")
    print(f"Run {run_id+1}/{NUM_RUNS} | Proxy: {proxy} | Seed: {seed}")
    print(f"{'='*80}")

    cmd = [
        'python', '201_macronas_local_search.py',
        '--proxy', proxy,
        '--dataset', DATASET,
        '--max_evals', str(MAX_EVALS),
        '--num_searches', '20',
        '--seed', str(seed),
        '--output', str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            print(f"[SUCCESS] Completed")
            return True
        else:
            print(f"[ERROR] Return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded 2 hours")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def collect_results():
    """Collect all results"""
    print(f"\n{'='*80}")
    print("Collecting results...")
    print(f"{'='*80}\n")

    results = {}

    for pkl_file in RUN_DIR.glob('*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            parts = pkl_file.stem.split('__')
            if len(parts) != 3:
                continue

            proxy = parts[1]

            if 'best_arch_str' in data and 'best_proxy_score' in data:
                if proxy not in results:
                    results[proxy] = []

                # Get scores and accuracies from all solutions
                all_scores = []
                all_accs = []
                if 'all_results' in data and data['all_results']:
                    for res in data['all_results']:
                        if 'proxy_score' in res:
                            all_scores.append(res['proxy_score'])
                        if 'accuracy' in res and res['accuracy'] is not None:
                            all_accs.append(res['accuracy'])

                results[proxy].append({
                    'run': parts[0],
                    'seed': parts[2],
                    'proxy_score': data['best_proxy_score'],
                    'params': data.get('best_params'),
                    'arch_str': data['best_arch_str'],
                    'accuracy': data.get('best_accuracy'),
                    'all_scores': all_scores,
                    'all_accs': all_accs
                })
        except Exception as e:
            print(f"[WARNING] Failed to load {pkl_file}: {e}")

    return results


def analyze_and_report(results):
    """Analyze results and generate report"""
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY ({DATASET})")
    print(f"{'='*80}\n")

    summary = []

    for proxy in PROXIES:
        if proxy not in results or len(results[proxy]) == 0:
            print(f"  {proxy:25s} - No results")
            continue

        scores = [r['proxy_score'] for r in results[proxy] if r['proxy_score'] is not None]
        accs = [r['accuracy'] for r in results[proxy] if r.get('accuracy') is not None]

        if len(scores) == 0:
            print(f"  {proxy:25s} - No valid scores")
            continue

        top1_score = max(scores)
        mean_top1 = np.mean(scores)
        std_top1 = np.std(scores)

        # Accuracy statistics
        if accs:
            top1_acc = max(accs)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
        else:
            top1_acc = None
            mean_acc = None
            std_acc = None

        # Mean score and accuracy of all solutions
        all_scores = []
        all_accs = []
        for r in results[proxy]:
            if 'all_scores' in r and r['all_scores']:
                all_scores.extend(r['all_scores'])
            if 'all_accs' in r and r['all_accs']:
                all_accs.extend(r['all_accs'])

        if all_scores:
            mean_all = np.mean(all_scores)
            std_all = np.std(all_scores)
        else:
            mean_all = None
            std_all = None

        if all_accs:
            mean_all_acc = np.mean(all_accs)
            std_all_acc = np.std(all_accs)
        else:
            mean_all_acc = None
            std_all_acc = None

        top1_run = max(results[proxy], key=lambda x: x['proxy_score'] if x['proxy_score'] else -1)

        summary.append({
            'proxy': proxy,
            'top1_score': top1_score,
            'mean_top1': mean_top1,
            'std_top1': std_top1,
            'top1_acc': top1_acc,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_all': mean_all,
            'std_all': std_all,
            'mean_all_acc': mean_all_acc,
            'std_all_acc': std_all_acc,
            'num_runs': len(scores),
            'num_solutions': len(all_scores),
            'top1_arch': top1_run['arch_str']
        })

        if top1_acc is not None:
            print(f"  {proxy:25s} - Top1Acc: {top1_acc:.2f}%  MeanAcc: {mean_acc:.2f}%±{std_acc:.2f}  ProxyScore: {top1_score:.2e}")
        else:
            print(f"  {proxy:25s} - ProxyScore: {top1_score:.2e}  MeanScore: {mean_top1:.2e}±{std_top1:.2e}")

    summary.sort(key=lambda x: x['top1_acc'] if x.get('top1_acc') else 0, reverse=True)

    print(f"\n  Ranking by Top1 Accuracy:")
    for i, res in enumerate(summary, 1):
        if res.get('top1_acc'):
            print(f"    {i}. {res['proxy']:25s} - Acc: {res['top1_acc']:.2f}%")

    summary_file = RUN_DIR / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

    return summary


def main():
    print(f"\n{'='*80}")
    print(f"NAS-201 MacroNAS Multi-Run Experiment")
    print(f"{'='*80}")
    print(f"Dataset: {DATASET}")
    print(f"Proxies: {len(PROXIES)}")
    print(f"Runs per proxy: {NUM_RUNS}")
    print(f"Max evaluations: {MAX_EVALS}")
    print(f"Total experiments: {len(PROXIES) * NUM_RUNS}")
    print(f"Results directory: {RUN_DIR}")

    # Check all historical directories for completed experiments
    all_existing = set()
    for run_dir in RESULTS_DIR.glob(f'run_{DATASET}_*'):
        for pkl in run_dir.glob('*.pkl'):
            all_existing.add(pkl.name)
    print(f"Found {len(all_existing)} existing results across all runs")
    print(f"{'='*80}\n")

    total_experiments = 0
    successful_experiments = 0

    for run_id in range(NUM_RUNS):
        seed = BASE_SEED + run_id

        print(f"\n{'#'*80}")
        print(f"# RUN {run_id+1}/{NUM_RUNS} (Seed: {seed})")
        print(f"{'#'*80}")

        for proxy in PROXIES:
            total_experiments += 1

            success = run_single_experiment(run_id, proxy, seed)
            if success:
                successful_experiments += 1

        print(f"Progress: {successful_experiments}/{total_experiments} successful")

    print(f"\n{'='*80}")
    print(f"All experiments completed: {successful_experiments}/{total_experiments} successful")
    print(f"{'='*80}")

    results = collect_results()
    summary = analyze_and_report(results)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved in: {RUN_DIR}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
