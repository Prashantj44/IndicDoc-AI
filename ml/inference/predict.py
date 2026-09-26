import torch
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def load_model(model_path: str, model_type: str, device: str = 'cpu') -> Any:
    """Load Faster R-CNN or YOLOv8 model."""
    if model_type.lower() == 'fasterrcnn':
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        return model
    elif model_type.lower() == 'yolov8':
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        return model
    else:
        raise ValueError("Unsupported model type")

def predict_single(model: Any, image: Any, model_type: str, device: str = 'cpu') -> Dict:
    """Predict on a single image."""
    if model_type.lower() == 'fasterrcnn':
        if isinstance(image, Image.Image):
            import torchvision.transforms as T
            image = T.ToTensor()(image)
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]
        return {
            'boxes': output['boxes'].cpu().numpy(),
            'labels': output['labels'].cpu().numpy(),
            'scores': output['scores'].cpu().numpy()
        }
    elif model_type.lower() == 'yolov8':
        results = model(image)[0]
        boxes = results.boxes
        return {
            'boxes': boxes.xyxy.cpu().numpy(),
            'labels': boxes.cls.cpu().numpy(),
            'scores': boxes.conf.cpu().numpy()
        }
    return {}

def predict_batch(model: Any, images: List, model_type: str, device: str = 'cpu') -> List[Dict]:
    """Predict on a batch of images."""
    return [predict_single(model, img, model_type, device) for img in images]

def measure_inference_time(model: Any, image: Any, model_type: str, num_runs: int = 10, warmup: int = 3, device: str = 'cpu') -> float:
    """Measure average inference time."""
    for _ in range(warmup):
        predict_single(model, image, model_type, device)
        
    start_time = time.time()
    for _ in range(num_runs):
        predict_single(model, image, model_type, device)
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

def save_annotated_image(image: Any, predictions: Dict, save_path: str) -> None:
    """Save image with drawn predictions."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
        
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box, label, score in zip(predictions.get('boxes', []), predictions.get('labels', []), predictions.get('scores', [])):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"cls {int(label)}: {score:.2f}", color='red', fontsize=8)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
