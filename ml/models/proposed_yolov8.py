import os
import json
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def get_proposed_model(model_size: str = 'n'):
    """
    Initialize YOLOv8 model for fine-tuning.
    Args:
        model_size (str): 'n', 's', 'm', 'l', or 'x' for YOLOv8 model sizes.
    """
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")
    
    # Load a pretrained YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    return model

def coco_to_yolo(coco_json_path: str, output_dir: str):
    """
    Converts COCO format annotations to YOLO format txt files.
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping image id to file
    img_dict = {img['id']: img for img in coco['images']}
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id'] # Note: YOLO classes are 0-indexed. Adjust if necessary.
        img_info = img_dict[img_id]
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        # COCO: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann['bbox']
        
        # YOLO: [x_center, y_center, width, height] (normalized)
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        txt_filename = Path(img_info['file_name']).stem + '.txt'
        txt_path = output_dir / txt_filename
        
        with open(txt_path, 'a', encoding='utf-8') as txt_f:
            txt_f.write(f"{cat_id} {x_center} {y_center} {norm_w} {norm_h}\n")
