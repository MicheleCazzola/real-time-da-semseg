import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from src.utils.variables import ModelType
from src.utils.variables import num_classes

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
    
    
#### IoU
def calculate_iou(outputs, masks, num_classes):

    # Get predictions from the model output probabilities
    _, preds = torch.max(outputs, dim=1) # B x H x W

    # IoU for each class
    iou_per_class = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)

    for i in range(num_classes):  # Iterate over all classes
        pred_mask = preds == i
        label_mask = masks == i

        intersection = torch.logical_and(pred_mask, label_mask).sum().float()
        union = torch.logical_or(pred_mask, label_mask).sum().float()

        if union > 0:
            iou_per_class[i] = intersection / union

    # Calculate mIoU for classes with a non-zero IoU
    valid_ious = iou_per_class
    miou = valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0, device=outputs.device)

    return miou, iou_per_class


#### Latency, FPS
def calculate_latency_fps(model, device, height, width, iterations, model_type: ModelType):
    image = torch.randn(1, 3, height, width).to(device)
    mask = None
    boundary = None

    if model_type == ModelType.PIDNET:
        mask = torch.randint(0, num_classes, (1, height, width), dtype=torch.int64).to(device)
        boundary = torch.randint(0, 2, (1, height, width), dtype=torch.float64).to(device)

    latency = []
    FPS = []

    for _ in range(iterations):
        start = time.time()

        with torch.no_grad():
            if model_type == ModelType.DEEPLAB:
                _ = model(image)
            else:
                _ = model(image, mask, boundary)

        end = time.time()

        latency_i = end - start
        latency.append(latency_i)

        FPS_i = 1 / latency_i
        FPS.append(FPS_i)

    meanLatency = np.mean(latency) * 1000 # millis
    stdLatency = np.std(latency) * 1000
    meanFPS = np.mean(FPS)
    stdFPS = np.std(FPS)

    return meanLatency, stdLatency, meanFPS, stdFPS

#### FLOPS, Params
def calculate_flops_params(model, device, height, width, model_type: ModelType):
    image = torch.zeros(1, 3, height, width).to(device)
    model = model.to(device)
    flops = None
    if model_type == ModelType.PIDNET:
        mask = torch.zeros(1, height, width, dtype=torch.int64).to(device)
        boundary = torch.zeros(1, height, width, dtype=torch.float64).to(device)
        flops = FlopCountAnalysis(model, (image, mask, boundary))
    else:
        flops = FlopCountAnalysis(model, image)
    
    return flop_count_table(flops)
    
    
# Performance metrics
def compute_performance_metrics(model, device, height, width, num_epochs, model_type: ModelType):
    model.eval()
    mean_latency, _, _, _ = calculate_latency_fps(model, device, height, width, num_epochs, model_type)

    flop_table = calculate_flops_params(model, device, height, width, model_type)
    
    return mean_latency, flop_table