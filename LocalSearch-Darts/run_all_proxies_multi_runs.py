"""
Batch run all proxies on NAS-301, 30 runs per proxy
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
    'ES', 'synflow', 'epsinas', 'naswot', 'MLFE',
    'epads', 'wrcor', 'zico', 'swap', 'dextr'
]

PROXIES = [
  'epsinas', 'meco', 'zen', 'near', 'ES', 'synflow', 'naswot', 'MLFE',
    'epads', 'wrcor', 'zico', 'swap', 'dextr'
]

NUM_RUNS = 30
MAX_EVALS = 500
BASE_SEED = 42

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'LocalSearch-Darts'
RESULTS_DIR.mkdir(exist_ok=True)

# Timestamp
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = RESULTS_DIR / f'run_{TIMESTAMP}'
RUN_DIR.mkdir(exist_ok=True)


def run_single_experiment(run_id, proxy, seed):
    """Run single experiment - using separate process"""
    output_file = RUN_DIR / f'run{run_id}__{proxy}__seed{seed}.pkl'
    
    # Check if already completed in current and historical directories
    filename = f'run{run_id}__{proxy}__seed{seed}.pkl'
    for run_dir in RESULTS_DIR.glob('run_*'):
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
        'python3', '301_macronas_local_search.py',
        '--proxy', proxy,
        '--max_evals', str(MAX_EVALS),
        '--num_searches', '20',
        '--seed', str(seed),
        '--output', str(output_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            timeout=7200
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
            
            if 'best_genotype' in data and 'best_proxy_score' in data:
                if proxy not in results:
                    results[proxy] = []
                
                results[proxy].append({
                    'run': parts[0],
                    'seed': parts[2],
                    'proxy_score': data['best_proxy_score'],
                    'params': data.get('best_params'),
                    'genotype': data['best_genotype'],
                    'accuracy': data.get('best_predicted_accuracy')
                })
        except Exception as e:
            print(f"[WARNING] Failed to load {pkl_file}: {e}")
    
    return results


def analyze_and_report(results):
    """Analyze results and generate report"""
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    summary = []
    
    for proxy in PROXIES:
        if proxy not in results or len(results[proxy]) == 0:
            print(f"  {proxy:25s} - No results")
            continue
        
        scores = [r['proxy_score'] for r in results[proxy] if r['proxy_score'] is not None]
        
        if len(scores) == 0:
            print(f"  {proxy:25s} - No valid scores")
            continue
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        top1_run = max(results[proxy], key=lambda x: x['proxy_score'] if x['proxy_score'] else -1)
        
        summary.append({
            'proxy': proxy,
            'mean_score': mean_score,
            'std_score': std_score,
            'max_score': max_score,
            'min_score': min_score,
            'num_runs': len(scores),
            'top1_genotype': top1_run['genotype'],
            'top1_score': top1_run['proxy_score']
        })
        
        print(f"  {proxy:25s} - Mean: {mean_score:.2e} ± {std_score:.2e} | Max: {max_score:.2e}")
    
    # Sort by max score
    summary.sort(key=lambda x: x['max_score'], reverse=True)
    
    print(f"\n{'='*80}")
    print("TOP PROXIES BY MAX SCORE:")
    print(f"{'='*80}")
    for i, res in enumerate(summary[:10], 1):
        print(f"  {i}. {res['proxy']:25s} - Max: {res['max_score']:.2e}")
    
    summary_file = RUN_DIR / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to {summary_file}")
    return summary


def main():
    print(f"{'='*80}")
    print("NAS-301 MacroNAS Multi-Run Experiment")
    print(f"{'='*80}")
    print(f"Proxies: {len(PROXIES)}")
    print(f"Runs per proxy: {NUM_RUNS}")
    print(f"Max evaluations: {MAX_EVALS}")
    print(f"Total experiments: {len(PROXIES) * NUM_RUNS}")
    print(f"Results directory: {RUN_DIR}")
    
    # Check all historical directories for completed experiments
    all_existing = set()
    for run_dir in RESULTS_DIR.glob('run_*'):
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
            
            print(f"\nProgress: {successful_experiments}/{total_experiments} successful")
            
            # Update summary after each experiment completion
            results = collect_results()
            summary = analyze_and_report(results)
    
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
