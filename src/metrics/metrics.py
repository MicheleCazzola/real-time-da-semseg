import torch

#### Average, Standard deviation
def compute_avg_std(dataloader, device):
    with torch.no_grad():
        avg = torch.zeros((1,3)).to(device)
        std = torch.zeros((1,3)).to(device)
        data_len = 0
        tot_pixels = 0

        assert len(dataloader) > 0, "Dataloader is empty"

        for inputs, labels in dataloader:
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