import json
import logging

import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from src.utils.variables import ModelType

#### Average, Standard deviation
def compute_avg_std(dataset, dataloader, device):
    with torch.no_grad():
        avg = torch.zeros((1,3)).to(device)
        std = torch.zeros((1,3)).to(device)
        data_len = 0
        tot_pixels = 0

        assert len(dataloader) > 0, "Dataloader must contain some data"

        tot_batches = len(dataloader)

        for (step, (inputs, labels)) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            b, _, h, w = inputs.shape

            data_len += b
            tot_pixels += b * h * w
            avg += torch.sum(inputs, dim=(0,2,3))
            std += torch.sum(inputs * inputs, dim=(0,2,3))

        avg /= tot_pixels
        std = torch.sqrt(std / tot_pixels - avg * avg)

        return data_len, avg.flatten().tolist(), std.flatten().tolist()
    
    
def compute_iou(outputs, masks, num_classes):

    # Avoid using argmax on MPS due to potential issues with autograd and NaN values
    if outputs.device.type == 'mps':
        preds = torch.argmax(outputs.cpu(), dim=1).to(outputs.device)
    else:
        preds = torch.argmax(outputs, dim=1)

    # IoU for each class
    ious = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)

    for i in range(num_classes):  # Iterate over all classes
        pred_mask = preds == i
        label_mask = masks == i

        intersection = torch.logical_and(pred_mask, label_mask).sum().float()
        union = torch.logical_or(pred_mask, label_mask).sum().float()

        if union > 0:
            ious[i] = intersection / union

    miou = ious.mean() if len(ious) > 0 else torch.tensor(0.0, device=outputs.device)

    return miou, ious

@torch.no_grad()
def calculate_fps_latency(model, device, iterations, num_classes, height, width, mask_required=False, bd_required=False):
    
    # Warmup
    warmpup_iterations = 100 if str(device) in ["cuda", "mps"] else 10
    for _ in range(warmpup_iterations):
        input_data = make_input(num_classes, height, width, mask_required, bd_required)
        input_data = tuple(d.to(device) for d in input_data)
        _ = model(*input_data)
        
    logging.info(f"Warmup completed with {warmpup_iterations} iterations on {device}")
        
    pool_size = 200
    input_pool = []
    for _ in range(pool_size):
        data = make_input(num_classes, height, width, mask_required, bd_required)
        input_pool.append(tuple(d.to(device) for d in data))
    
    if device == "cuda":
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(iterations):
            _ = model(*input_pool[i % pool_size])
        end_event.record()
        
        torch.cuda.synchronize()
        
        elapsed = start_event.elapsed_time(end_event) / 1000
    elif device == "mps":
        torch.mps.synchronize()
        
        start_time = time.perf_counter_ns()
        for i in range(iterations):
            _ = model(*input_pool[i % pool_size])
        
        torch.mps.synchronize()
        end_time = time.perf_counter_ns()
        
        elapsed = (end_time - start_time) / 1e9
    else:
        start_time = time.perf_counter_ns()
        for i in range(iterations):
            _ = model(*input_pool[i % pool_size])
        end_time = time.perf_counter_ns()
        
        elapsed = (end_time - start_time) / 1e9
        
    fps = iterations / elapsed
    mean_latency = 1 / fps
    
    logging.info(f"Computed FPS and average latency over {iterations} iterations")
    
    return fps, mean_latency

#### FLOPS, Params
def calculate_flops_params(model, device, num_classes, height, width, mask_required=False, bd_required=False):
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    input_data = make_input(num_classes, height, width, mask_required, bd_required)
    input_data = tuple(d.to(device) for d in input_data)
    flops = FlopCountAnalysis(model, input_data)
    
    logging.info(f"Computed FLOPs and parameter count")
    
    return num_params, flops.total(), flop_count_table(flops)
    
def make_input(num_classes, height, width, mask_required=False, bd_required=False):
    input_data = [torch.randn(1, 3, height, width)]
    if mask_required:
        input_data.append(torch.randint(0, num_classes, (1, height, width), dtype=torch.long))
    if bd_required:
        input_data.append(torch.randint(0, 2, (1, height, width), dtype=torch.float32))
    return tuple(input_data)
    
@torch.no_grad()
def compute_performance_metrics(model, num_classes, device, height, width, iterations, mask_required=False, bd_required=False, save_to=None, return_out=False):
    model.eval()
    
    fps, mean_latency = calculate_fps_latency(model, device, iterations, num_classes, height, width, mask_required=mask_required, bd_required=bd_required)

    # Needs CPU for FLOPs with fvcore
    num_params, total_flops, flop_table = calculate_flops_params(model, "cpu", num_classes, height, width, mask_required=mask_required, bd_required=bd_required)
    
    if save_to is not None:
        json_str = json.dumps({
            "fps": fps,
            "mean_latency": mean_latency,
            "num_params": num_params,
            "total_flops": total_flops,
            "flop_table": flop_table
        }, indent=4)
        with open(save_to, "w") as f:
            f.write(json_str)
    
    if return_out:
        return {
            "fps": fps,
            "mean_latency": mean_latency,
            "num_params": num_params,
            "total_flops": total_flops,
            "flop_table": flop_table
        }