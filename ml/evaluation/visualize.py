import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import json
import os
from typing import List, Dict, Tuple, Any

def draw_predictions(image: np.ndarray, boxes: List[List[float]], labels: List[str], scores: List[float], save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"{label}: {score:.2f}", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def draw_ground_truth(image: np.ndarray, boxes: List[List[float]], labels: List[str], save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"{label}", color='green', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def draw_comparison(image: np.ndarray, pred_boxes: List, pred_labels: List, gt_boxes: List, gt_labels: List, save_path: str = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image)
    ax1.set_title("Predictions")
    for box, label in zip(pred_boxes, pred_labels):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(box[0], box[1] - 5, f"{label}", color='red', fontsize=8)
        
    ax2.imshow(image)
    ax2.set_title("Ground Truth")
    for box, label in zip(gt_boxes, gt_labels):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(box[0], box[1] - 5, f"{label}", color='green', fontsize=8)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_precision_recall_curve(precisions: List[float], recalls: List[float], class_name: str, save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, label=f'PR Curve for {class_name}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_class_distribution(annotations_file: str, save_path: str = None) -> plt.Figure:
    with open(annotations_file, 'r') as f:
        data = json.load(f)
        
    class_counts = {}
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
        
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    names = [categories.get(cat_id, str(cat_id)) for cat_id in class_counts.keys()]
    counts = list(class_counts.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, counts)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequencies')
    plt.xticks(rotation=45)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig
