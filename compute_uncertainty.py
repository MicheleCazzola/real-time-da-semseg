import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import argparse

from src.dataset.dataset import LoveDA
from src.dataset.dataset import generate_bd

# Initialize constants and paths
NC = 7
NB = 20
ROOT = "/Volumes/TOSHIBA/real_time_da_semseg_eval_objects" # eval_objects

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
parser.add_argument("--root", type=str, default=ROOT, help="Root directory for evaluation objects")
parser.add_argument("--domain", type=str, required=True, help="Domain name")
args = parser.parse_args()

MODEL = args.model
ROOT = args.root
DOMAIN = args.domain

URBAN_GT, URBAN_PRED = os.path.join(ROOT, "urban", "gt"), os.path.join(ROOT, "urban", MODEL)
RURAL_GT, RURAL_PRED = os.path.join(ROOT, "rural", "gt"), os.path.join(ROOT, "rural", MODEL)

if DOMAIN == "urban":
    gt_folder = URBAN_GT
    pred_folder = URBAN_PRED
elif DOMAIN == "rural":
    gt_folder = RURAL_GT
    pred_folder = RURAL_PRED

# Inizialize dictionaries to store results
conf_per_class = {c: {"mean": 0, "m2": 0, "std": 0, "count": 0, "bins": [0] * NB} for c in range(NC)}
conf_per_outcome = {"correct": {"mean": 0, "m2": 0, "std": 0, "count": 0, "bins": [0] * NB}, "incorrect": {"mean": 0, "m2": 0, "std": 0, "count": 0, "bins": [0] * NB}}
binned_pixel_acc_per_class = {c: {"count": [0] * NB, "acc": [0] * NB, "conf": [0] * NB} for c in range(NC)}

roc_per_class = {c: {th: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for th in [i/NB for i in range(NB)]} for c in range(NC)}
prc_per_class = {c: {th: {"tp": 0, "fp": 0, "fn": 0} for th in [i/NB for i in range(NB)]} for c in range(NC)}
roc_overall = {th: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for th in [i/NB for i in range(NB)]}
prc_overall = {th: {"tp": 0, "fp": 0, "fn": 0} for th in [i/NB for i in range(NB)]}

conf_per_area = {e: {"mean": 0, "m2": 0, "std": 0, "count": 0, "bins": [0] * NB} for e in ["boundary", "near_boundary", "non_boundary"]}
binned_pixel_acc_per_area = {e: {"count": [0] * NB, "acc": [0] * NB, "conf": [0] * NB} for e in ["boundary", "near_boundary", "non_boundary"]}
roc_per_area = {e: {th: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for th in [i/NB for i in range(NB)]} for e in ["boundary", "near_boundary", "non_boundary"]}
prc_per_area = {e: {th: {"tp": 0, "fp": 0, "fn": 0} for th in [i/NB for i in range(NB)]} for e in ["boundary", "near_boundary", "non_boundary"]}

gt_list = sorted([x for x in os.listdir(gt_folder) if x.endswith(".pt") and not x.startswith(".")])
pred_list = sorted([x for x in os.listdir(pred_folder) if x.endswith(".pt") and not x.startswith(".")])
# gt_list = gt_list[:20]
# pred_list = pred_list[:20]

print(f"Processing {len(gt_list)} files from {gt_folder} and {pred_folder}...")

# Process each pair of ground truth and prediction files
for gt_file, pred_file in tqdm(zip(gt_list, pred_list), desc="Processing files", total=len(gt_list)):
    gt_path = os.path.join(gt_folder, gt_file)
    pred_path = os.path.join(pred_folder, pred_file)

    masks = torch.load(gt_path, weights_only=False).unsqueeze(0)
    normalized_pred = torch.load(pred_path, weights_only=False).unsqueeze(0)
    
    pred_class = normalized_pred.argmax(dim=1)
    pred_conf = normalized_pred.max(dim=1).values
    score = 1 - pred_conf
    positive_class_roc_prc = pred_class != masks
    
    # Edges and near-edges for boundary analysis
    edges_tight = torch.from_numpy(generate_bd(pred_class.squeeze(0).numpy().astype(np.uint8), iterations=1))
    edges_loose = torch.from_numpy(generate_bd(pred_class.squeeze(0).numpy().astype(np.uint8), iterations=10))
    boundaries = (edges_tight == 1) & (masks != -1)
    near_boundaries = (edges_loose == 1) & (edges_tight == 0) & (masks != -1)
    non_boundaries = (edges_tight == 0) & (edges_loose == 0) & (masks != -1)
    
    # if pred_file == "0000.pt":
    #     print(edges_tight.shape, pred_class.shape)
    #     print(f"File: {pred_file}, Total pixels: {masks.numel()}, Valid pixels: {masks.ne(-1).sum().item()}, Boundary pixels: {boundaries.sum().item()}, Near-boundary pixels: {near_boundaries.sum().item()}, Non-boundary pixels: {non_boundaries.sum().item()}")
    
    #     import matplotlib.pyplot as plt
    #     plt.imsave(f"{pred_file.replace('.pt', '')}_et.png", edges_tight.numpy(), cmap='grey')
    
    # Confidence per class
    for c in range(NC):
        class_mask = (masks == c)
        if class_mask.any():
            conf_target = normalized_pred[:, c, :, :][class_mask]
            conf_per_class[c]["mean"] += conf_target.sum().item()
            conf_per_class[c]["m2"] += (conf_target ** 2).sum().item()
            conf_per_class[c]["count"] += conf_target.numel()
            for bin in range(len(conf_per_class[c]["bins"])):
                conf_per_class[c]["bins"][bin] += ((bin/20 <= conf_target) & (conf_target < (bin + 1) / 20)).sum().item()
            del conf_target
        del class_mask
        
    # Confidence per outcome (correct vs incorrect)
    for outcome in ["correct", "incorrect"]:
        outcome_mask = (pred_class == masks) if outcome == "correct" else (pred_class != masks)
        if outcome_mask.any():
            conf_target = pred_conf[outcome_mask]
            conf_per_outcome[outcome]["mean"] += conf_target.sum().item()
            conf_per_outcome[outcome]["m2"] += (conf_target ** 2).sum().item()
            conf_per_outcome[outcome]["count"] += conf_target.numel()
            
            for bin in range(len(conf_per_outcome[outcome]["bins"])):
                conf_per_outcome[outcome]["bins"][bin] += ((bin/NB <= conf_target) & (conf_target < (bin + 1) / NB)).sum().item()
            del conf_target
        del outcome_mask
    
    # Pixel-wise accuracy per class and bin (will be used later to compute ECE and MCE)
    for c in range(NC):
        class_mask = (pred_class == c)
        
        if class_mask.any():
            correct_mask = (pred_class == masks)
            histograms = [
                ((bin/NB <= pred_conf) & (pred_conf < (bin + 1) / NB))
                for bin in range(len(binned_pixel_acc_per_class[c]["acc"])) 
            ]
            
            # Accuracy per bin, called acc(B_m)
            # Counts the number of correct predictions for each pixel in the bin
            binned_pixel_acc_per_class[c]["acc"] = [
                binned_pixel_acc_per_class[c]["acc"][bin] + (correct_mask & class_mask & histograms[bin]).sum().item()
                for bin in range(len(binned_pixel_acc_per_class[c]["acc"]))
            ]
            
            # Confidence per bin, called conf(B_m)
            # Sums all the confidence values for each pixel in the bin, regardless of whether they are correct or incorrect
            binned_pixel_acc_per_class[c]["conf"] = [
                binned_pixel_acc_per_class[c]["conf"][bin] + (pred_conf[class_mask & histograms[bin]].sum().item())    
                for bin in range(len(binned_pixel_acc_per_class[c]["conf"]))
            ]
            
            # Count of pixels for this class for each bin, called |B_m|
            binned_pixel_acc_per_class[c]["count"] = [
                binned_pixel_acc_per_class[c]["count"][bin] + (class_mask & histograms[bin]).sum().item()
                for bin in range(len(binned_pixel_acc_per_class[c]["count"]))
            ]
            del histograms
            del correct_mask
        del class_mask
    
    # ROC and PRC per class
    for c in range(NC):
        for th in [i/NB for i in range(NB)]:
            high_uncertainty = (score >= th)
            
            true_positive = (high_uncertainty & positive_class_roc_prc & (masks == c)).sum().item()
            false_positive = (high_uncertainty & ~positive_class_roc_prc & (masks == c)).sum().item()
            true_negative = (~high_uncertainty & ~positive_class_roc_prc & (masks == c)).sum().item()
            false_negative = (~high_uncertainty & positive_class_roc_prc & (masks == c)).sum().item()
            
            roc_per_class[c][th]["tp"] += true_positive
            roc_per_class[c][th]["fp"] += false_positive
            roc_per_class[c][th]["tn"] += true_negative
            roc_per_class[c][th]["fn"] += false_negative
            
            prc_per_class[c][th]["tp"] += true_positive
            prc_per_class[c][th]["fp"] += false_positive
            prc_per_class[c][th]["fn"] += false_negative
            
            del high_uncertainty
    
    # AUROC and AUPRC overall
    for th in [i/NB for i in range(NB)]:
        high_uncertainty = (score >= th)
        
        true_positive = (high_uncertainty & positive_class_roc_prc & (masks != -1)).sum().item()
        false_positive = (high_uncertainty & ~positive_class_roc_prc & (masks != -1)).sum().item()
        true_negative = (~high_uncertainty & ~positive_class_roc_prc & (masks != -1)).sum().item()
        false_negative = (~high_uncertainty & positive_class_roc_prc & (masks != -1)).sum().item()
        
        roc_overall[th]["tp"] += true_positive
        roc_overall[th]["fp"] += false_positive
        roc_overall[th]["tn"] += true_negative
        roc_overall[th]["fn"] += false_negative
        
        prc_overall[th]["tp"] += true_positive
        prc_overall[th]["fp"] += false_positive
        prc_overall[th]["fn"] += false_negative
        
        del high_uncertainty
    
    # Boundary analysis: confidence per area (boundary, near-boundary, non-boundary)
    for area, area_mask in zip(["boundary", "near_boundary", "non_boundary"], [boundaries, near_boundaries, non_boundaries]):
        if area_mask.any():
            conf_target = pred_conf[area_mask]
            conf_per_area[area]["mean"] += conf_target.sum().item()
            conf_per_area[area]["m2"] += (conf_target ** 2).sum().item()
            conf_per_area[area]["count"] += conf_target.numel()
            
            # if pred_file == "0000.pt":
            #     print(f"Area: {area}, Count: {conf_target.numel()}, Mean: {conf_target.mean().item():.4f}, Std: {conf_target.std().item():.4f}")
            
            for bin in range(len(conf_per_area[area]["bins"])):
                conf_per_area[area]["bins"][bin] += ((bin/NB <= conf_target) & (conf_target < (bin + 1) / NB)).sum().item()
            del conf_target
        del area_mask
    
    # Boundary analysis: pixel-wise accuracy per area and bin
    for area, area_mask in zip(["boundary", "near_boundary", "non_boundary"], [boundaries, near_boundaries, non_boundaries]):
        if area_mask.any():
            correct_mask = (pred_class == masks)
            histograms = [
                ((bin/NB <= pred_conf) & (pred_conf < (bin + 1) / NB))
                for bin in range(len(binned_pixel_acc_per_area[area]["acc"])) 
            ]
            
            # Accuracy per bin, called acc(B_m)
            binned_pixel_acc_per_area[area]["acc"] = [
                binned_pixel_acc_per_area[area]["acc"][bin] + (correct_mask & area_mask & histograms[bin]).sum().item()
                for bin in range(len(binned_pixel_acc_per_area[area]["acc"]))
            ]
            
            # Confidence per bin, called conf(B_m)
            binned_pixel_acc_per_area[area]["conf"] = [
                binned_pixel_acc_per_area[area]["conf"][bin] + (pred_conf[area_mask & histograms[bin]].sum().item())    
                for bin in range(len(binned_pixel_acc_per_area[area]["conf"]))
            ]
            
            # Count of pixels for this area for each bin, called |B_m|
            binned_pixel_acc_per_area[area]["count"] = [
                binned_pixel_acc_per_area[area]["count"][bin] + (area_mask & histograms[bin]).sum().item()
                for bin in range(len(binned_pixel_acc_per_area[area]["count"]))
            ]
            del histograms
            del correct_mask
        del area_mask
    
    # Boundary analysis: ROC and PRC per area
    for area, area_mask in zip(["boundary", "near_boundary", "non_boundary"], [boundaries, near_boundaries, non_boundaries]):
        for th in [i/NB for i in range(NB)]:
            high_uncertainty = (score >= th)
            
            true_positive = (high_uncertainty & positive_class_roc_prc & area_mask).sum().item()
            false_positive = (high_uncertainty & ~positive_class_roc_prc & area_mask).sum().item()
            true_negative = (~high_uncertainty & ~positive_class_roc_prc & area_mask).sum().item()
            false_negative = (~high_uncertainty & positive_class_roc_prc & area_mask).sum().item()
            
            roc_per_area[area][th]["tp"] += true_positive
            roc_per_area[area][th]["fp"] += false_positive
            roc_per_area[area][th]["tn"] += true_negative
            roc_per_area[area][th]["fn"] += false_negative
            
            prc_per_area[area][th]["tp"] += true_positive
            prc_per_area[area][th]["fp"] += false_positive
            prc_per_area[area][th]["fn"] += false_negative
            
            del high_uncertainty
    
    del masks
    del normalized_pred
    

# Finalize the results by computing means, standard deviations, and normalizing histograms

new_conf_per_class = {}
# Confidence per class
for c in range(NC):
    if conf_per_class[c]["count"] > 0:
        conf_per_class[c]["mean"] = conf_per_class[c]["mean"] / conf_per_class[c]["count"]
        conf_per_class[c]["m2"] = conf_per_class[c]["m2"] / conf_per_class[c]["count"]
        conf_per_class[c]["std"] = (conf_per_class[c]["m2"] - (conf_per_class[c]["mean"] ** 2)) ** 0.5
        conf_per_class[c]["bins"] = [b / conf_per_class[c]["count"] for b in conf_per_class[c]["bins"]]
    else:
        conf_per_class[c]["mean"] = 0
        conf_per_class[c]["std"] = 0
        conf_per_class[c]["bins"] = [0] * NB
        conf_per_class[c]["m2"] = 0
    new_conf_per_class[LoveDA.id2label[c]] = conf_per_class[c]

# Confidence per outcome
for outcome in ["correct", "incorrect"]:
    if conf_per_outcome[outcome]["count"] > 0:
        conf_per_outcome[outcome]["mean"] = conf_per_outcome[outcome]["mean"] / conf_per_outcome[outcome]["count"]
        conf_per_outcome[outcome]["m2"] = conf_per_outcome[outcome]["m2"] / conf_per_outcome[outcome]["count"]
        conf_per_outcome[outcome]["std"] = ((conf_per_outcome[outcome]["m2"] - (conf_per_outcome[outcome]["mean"] ** 2)) ** 0.5) if conf_per_outcome[outcome]["count"] > 0 else 0
        conf_per_outcome[outcome]["bins"] = [b / conf_per_outcome[outcome]["count"] for b in conf_per_outcome[outcome]["bins"]]
    else:
        conf_per_outcome[outcome]["mean"] = 0
        conf_per_outcome[outcome]["std"] = 0
        conf_per_outcome[outcome]["bins"] = [0] * NB
        conf_per_outcome[outcome]["m2"] = 0
        
# Pixel-wise accuracy per class and bin
new_binned_pixel_acc_per_class = {}
for c in range(NC):
    if sum(binned_pixel_acc_per_class[c]["count"]) > 0:
        binned_pixel_acc_per_class[c]["acc"] = [(acc / count if count > 0 else 0) for acc, count in zip(binned_pixel_acc_per_class[c]["acc"], binned_pixel_acc_per_class[c]["count"])]
        binned_pixel_acc_per_class[c]["conf"] = [(conf / count if count > 0 else 0) for conf, count in zip(binned_pixel_acc_per_class[c]["conf"], binned_pixel_acc_per_class[c]["count"])]
        binned_pixel_acc_per_class[c]["count"] = [count for count in binned_pixel_acc_per_class[c]["count"]]
    else:
        binned_pixel_acc_per_class[c]["acc"] = [0] * NB
        binned_pixel_acc_per_class[c]["conf"] = [0] * NB
        binned_pixel_acc_per_class[c]["count"] = [0] * NB
        
    new_binned_pixel_acc_per_class[LoveDA.id2label[c]] = binned_pixel_acc_per_class[c]
    
# ROC and PRC per class
new_roc_per_class = {}
new_prc_per_class = {}
for c in range(NC):
    for th in roc_per_class[c]:
        
        tp, fp, tn, fn = roc_per_class[c][th]["tp"], roc_per_class[c][th]["fp"], roc_per_class[c][th]["tn"], roc_per_class[c][th]["fn"]
        roc_per_class[c][th]["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_per_class[c][th]["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        prc_per_class[c][th]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        prc_per_class[c][th]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    new_roc_per_class[LoveDA.id2label[c]] = roc_per_class[c]
    new_prc_per_class[LoveDA.id2label[c]] = prc_per_class[c]

# ROC and PRC overall
for th in roc_overall:
    
    tp, fp, tn, fn = roc_overall[th]["tp"], roc_overall[th]["fp"], roc_overall[th]["tn"], roc_overall[th]["fn"]
    roc_overall[th]["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    roc_overall[th]["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    prc_overall[th]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    prc_overall[th]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0

# Boundary analysis: confidence per area
new_conf_per_area = {}
for area in ["boundary", "near_boundary", "non_boundary"]:
    if conf_per_area[area]["count"] > 0:
        conf_per_area[area]["mean"] = conf_per_area[area]["mean"] / conf_per_area[area]["count"]
        conf_per_area[area]["m2"] = conf_per_area[area]["m2"] / conf_per_area[area]["count"]
        conf_per_area[area]["std"] = (conf_per_area[area]["m2"] - (conf_per_area[area]["mean"] ** 2)) ** 0.5
        conf_per_area[area]["bins"] = [b / conf_per_area[area]["count"] for b in conf_per_area[area]["bins"]]
    else:
        conf_per_area[area]["mean"] = 0
        conf_per_area[area]["std"] = 0
        conf_per_area[area]["bins"] = [0] * NB
        conf_per_area[area]["m2"] = 0
    new_conf_per_area[area] = conf_per_area[area]

# Boundary analysis: pixel-wise accuracy per area and bin
new_binned_pixel_acc_per_area = {}
for area in ["boundary", "near_boundary", "non_boundary"]:
    if sum(binned_pixel_acc_per_area[area]["count"]) > 0:
        binned_pixel_acc_per_area[area]["acc"] = [(acc / count if count > 0 else 0) for acc, count in zip(binned_pixel_acc_per_area[area]["acc"], binned_pixel_acc_per_area[area]["count"])]
        binned_pixel_acc_per_area[area]["conf"] = [(conf / count if count > 0 else 0) for conf, count in zip(binned_pixel_acc_per_area[area]["conf"], binned_pixel_acc_per_area[area]["count"])]
        binned_pixel_acc_per_area[area]["count"] = [count for count in binned_pixel_acc_per_area[area]["count"]]
    else:
        binned_pixel_acc_per_area[area]["acc"] = [0] * NB
        binned_pixel_acc_per_area[area]["conf"] = [0] * NB
        binned_pixel_acc_per_area[area]["count"] = [0] * NB
        
    new_binned_pixel_acc_per_area[area] = binned_pixel_acc_per_area[area]

# Boundary analysis: ROC and PRC per area
new_roc_per_area = {}
new_prc_per_area = {}
for area in ["boundary", "near_boundary", "non_boundary"]:
    for th in roc_per_area[area]:
        
        tp, fp, tn, fn = roc_per_area[area][th]["tp"], roc_per_area[area][th]["fp"], roc_per_area[area][th]["tn"], roc_per_area[area][th]["fn"]
        roc_per_area[area][th]["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_per_area[area][th]["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        prc_per_area[area][th]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        prc_per_area[area][th]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    new_roc_per_area[area] = roc_per_area[area]
    new_prc_per_area[area] = prc_per_area[area]

# Save results to JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_data = {
    "per_class": new_conf_per_class,
    "per_outcome": conf_per_outcome,
    "binned_pixel_acc_per_class": new_binned_pixel_acc_per_class,
    "roc_per_class": new_roc_per_class,
    "prc_per_class": new_prc_per_class,
    "roc_overall": roc_overall,
    "prc_overall": prc_overall,
    "per_area": new_conf_per_area,
    "binned_pixel_acc_per_area": new_binned_pixel_acc_per_area,
    "roc_per_area": new_roc_per_area,
    "prc_per_area": new_prc_per_area
}

path = os.path.join("outputs_uncertainty", MODEL, timestamp)
os.makedirs(path, exist_ok=True)
output_file = os.path.join(path, "uq_infos.json")
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)