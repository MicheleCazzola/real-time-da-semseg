import logging
import json
import time
import torch
import numpy as np

from fvcore.nn import FlopCountAnalysis, flop_count_table

def run_warmup(model, device, height, width):
    iterations = 100 if str(device) in ["cuda", "mps"] else 10
    for _ in range(iterations):
        input_tensor = torch.randn(1, 3, height, width, device=device)
        _ = model(input_tensor)
        
    logging.info(f"Warmup completed with {iterations} iterations on {device}")
    
def make_input_pool(device, height, width, pool_size=200):
    input_pool = []
    for _ in range(pool_size):
        input_tensor = torch.randn(1, 3, height, width, device=device)
        input_pool.append(input_tensor)
    return input_pool

@torch.no_grad()
def compute_fps(model, device, iterations, height, width):
    
    # Warmup
    run_warmup(model, device, height, width)
    
    # Use a pool of inputs to avoid caching effects
    pool_size = 200
    input_pool = make_input_pool(device, height, width, pool_size=pool_size)
    
    if device == "cuda":
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(iterations):
            _ = model(input_pool[i % pool_size])
        end_event.record()
        
        torch.cuda.synchronize()
        
        elapsed = start_event.elapsed_time(end_event) / 1000
        
    elif device == "mps":
        torch.mps.synchronize()
        start_time = time.perf_counter_ns()
        
        for i in range(iterations):
            _ = model(input_pool[i % pool_size])
        
        torch.mps.synchronize()
        end_time = time.perf_counter_ns()
        
        elapsed = (end_time - start_time) / 1e9
    else:
        start_time = time.perf_counter_ns()
        for i in range(iterations):
            _ = model(input_pool[i % pool_size])
        end_time = time.perf_counter_ns()
        
        elapsed = (end_time - start_time) / 1e9
        
    fps = iterations / elapsed
    
    logging.info(f"Computed FPS over {iterations} iterations: {fps:.2f} on {device}")
    
    return fps

@torch.no_grad()
def compute_latency(model, device, iterations, height, width):
    
    # Warmup
    run_warmup(model, device, height, width)
        
    # Use a pool of inputs to avoid caching effects
    pool_size = 200
    input_pool = make_input_pool(device, height, width, pool_size=pool_size)
        
    latencies = []
    for i in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model(input_pool[i % pool_size])
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000
        elif device == "mps":
            torch.mps.synchronize()
            
            start_time = time.perf_counter_ns()
            _ = model(input_pool[i % pool_size])
            torch.mps.synchronize()
            end_time = time.perf_counter_ns()
            
            elapsed = (end_time - start_time) / 1e9
        else:
            start_time = time.perf_counter_ns()
            _ = model(input_pool[i % pool_size])
            end_time = time.perf_counter_ns()
            
            elapsed = (end_time - start_time) / 1e9
        
        latencies.append(elapsed)
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    logging.info(f"Computed latency over {iterations} iterations: Mean = {mean_latency:.4f} s, Std = {std_latency:.4f} s on {device}")
    
    return float(mean_latency), float(std_latency)

@torch.no_grad()
def compute_num_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Computed number of parameters: {num_params}")
    return num_params

@torch.no_grad()
def compute_flops(model, height, width, device="cpu"):
    model = model.to(device)
    
    input = torch.randn(1, 3, height, width, device=device)
    flops = FlopCountAnalysis(model, input)
    
    logging.info(f"Computed FLOPs")
    
    return flops.total(), flop_count_table(flops)
    
@torch.no_grad()
def compute_performance_metrics(model, iterations, height, width, device, save_to=None, return_out=False):
    model.eval()
    
    fps = compute_fps(model, device, iterations, height, width)
    mean_latency, std_latency = compute_latency(model, device, iterations, height, width)
    
    num_params = compute_num_params(model)
    
    # Needs CPU for FLOPs with fvcore
    total_flops, flop_table = compute_flops(model, height, width)
    
    performance_result = {
        "fps": fps,
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "num_params": num_params,
        "total_flops": total_flops,
        "flop_table": flop_table 
    }
    
    if save_to is not None:
        json_str = json.dumps(performance_result, indent=4)
        with open(save_to, "w") as f:
            f.write(json_str)
    
    if return_out:
        return performance_result