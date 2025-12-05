import os
import sys
import time
import torch
import numpy as np

# Add stylegan2_ada_pytorch to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))

import dnnlib
import legacy

def benchmark(network_pkl, num_samples=100, device='cuda'):
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print(f'Running benchmark with {num_samples} samples...')
    
    # Warmup
    print("Warming up...")
    z = torch.randn([1, G.z_dim], device=device)
    c = None
    for _ in range(10):
        _ = G(z, c)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    latencies = []
    for i in range(num_samples):
        z = torch.randn([1, G.z_dim], device=device)
        t0 = time.time()
        _ = G(z, c)
        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms

    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    fps = 1.0 / (avg_latency / 1000.0)
    
    print(f"\nResults ({num_samples} images):")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Average Latency: {avg_latency:.2f} ms/image")
    print(f"Throughput: {fps:.2f} FPS")
    
    return avg_latency, fps

if __name__ == "__main__":
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    # Check if local file exists to save download time
    if os.path.exists("checkpoints/ffhq.pkl"):
        network_pkl = "checkpoints/ffhq.pkl"
        
    benchmark(network_pkl)
