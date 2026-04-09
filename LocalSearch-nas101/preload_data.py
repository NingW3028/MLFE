"""Run this in tf environment to preload NAS-101 data"""
from nasbench import api
import pickle
import numpy as np
import os

print("Loading NAS-101...")
NASBENCH_TFRECORD = os.path.join('..', 'NAS101', 'nasbench_only108.tfrecord')
NASBENCH_TFRECORD = os.path.abspath(NASBENCH_TFRECORD)
print(f"Loading from: {NASBENCH_TFRECORD}")
nasbench = api.NASBench(NASBENCH_TFRECORD)

print("Extracting architecture data...")
data_cache = {}
count = 0

for unique_hash in nasbench.hash_iterator():
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
    
    matrix = fixed_metrics['module_adjacency']
    ops = fixed_metrics['module_operations']
    
    # Get validation accuracy (average over epochs)
    accuracy = np.mean([computed_metrics[108][i]['final_validation_accuracy'] 
                       for i in range(3)])
    params = fixed_metrics['trainable_parameters']
    
    key = (tuple(map(tuple, matrix)), tuple(ops))
    data_cache[key] = {'accuracy': accuracy, 'params': params}
    
    count += 1
    if count % 10000 == 0:
        print(f"Processed {count} architectures...")

print(f"Total architectures: {len(data_cache)}")
print("Saving cache...")

with open('nas101_cache.pkl', 'wb') as f:
    pickle.dump(data_cache, f)

print("Done! Cache saved to nas101_cache.pkl")
