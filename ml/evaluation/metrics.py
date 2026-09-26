import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area

def calculate_ap(precision, recall) -> float:
    """Calculate Average Precision using 11-point interpolation."""
    precision = np.array(precision)
    recall = np.array(recall)
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.
    return ap

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str, num_classes: int, iou_threshold: float = 0.5) -> Dict:
    """Evaluate Faster R-CNN model on a dataloader."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for out, target in zip(outputs, targets):
                pred_boxes = out['boxes'].cpu().numpy()
                pred_labels = out['labels'].cpu().numpy()
                pred_scores = out['scores'].cpu().numpy()
                
                target_boxes = target['boxes'].cpu().numpy()
                target_labels = target['labels'].cpu().numpy()
                
                for i in range(len(pred_boxes)):
                    all_preds.append({
                        'box': pred_boxes[i],
                        'label': pred_labels[i],
                        'score': pred_scores[i]
                    })
                    
                for i in range(len(target_boxes)):
                    all_targets.append({
                        'box': target_boxes[i],
                        'label': target_labels[i]
                    })
                    
    results = {}
    for c in range(1, num_classes):
        c_preds = sorted([p for p in all_preds if p['label'] == c], key=lambda x: x['score'], reverse=True)
        c_targets = [t for t in all_targets if t['label'] == c]
        
        if len(c_targets) == 0:
            continue
            
        tp = np.zeros(len(c_preds))
        fp = np.zeros(len(c_preds))
        matched = set()
        
        for i, pred in enumerate(c_preds):
            best_iou = 0
            best_target_idx = -1
            for j, target in enumerate(c_targets):
                if j in matched:
                    continue
                iou = calculate_iou(pred['box'], target['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched.add(best_target_idx)
            else:
                fp[i] = 1
                
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        
        recalls = tp_cumsum / len(c_targets) if len(c_targets) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        ap = calculate_ap(precisions, recalls)
        
        results[f'class_{c}_ap'] = ap
        if len(precisions) > 0:
            results[f'class_{c}_precision'] = precisions[-1]
            results[f'class_{c}_recall'] = recalls[-1]
            
    ap_values = [v for k, v in results.items() if k.endswith('_ap')]
    results['mAP'] = float(np.mean(ap_values)) if ap_values else 0.0
    
    return results

def save_metrics_report(metrics: Dict, filepath: str) -> None:
    """Save metrics to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def generate_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int] = None) -> np.ndarray:
    """Generate confusion matrix using sklearn."""
    return confusion_matrix(y_true, y_pred, labels=labels)
