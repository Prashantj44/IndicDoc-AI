import os
import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.config import *
from ml.models.proposed_yolov8 import get_proposed_model, coco_to_yolo

def create_yolo_yaml(train_img_dir: str, val_img_dir: str, num_classes: int, class_names: list, output_path: str):
    data = {
        'train': str(train_img_dir),
        'val': str(val_img_dir),
        'nc': num_classes,
        'names': class_names
    }
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main(args):
    # Setup directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert COCO to YOLO if needed
    if args.convert_annotations:
        print("Converting COCO annotations to YOLO format...")
        train_label_dir = Path(args.train_img).parent / 'labels'
        val_label_dir = Path(args.val_img).parent / 'labels'
        coco_to_yolo(args.train_ann, str(train_label_dir))
        coco_to_yolo(args.val_ann, str(val_label_dir))
    
    # 2. Create YAML
    yaml_path = MODEL_DIR / 'yolo_dataset.yaml'
    create_yolo_yaml(args.train_img, args.val_img, NUM_CLASSES, CLASS_NAMES, str(yaml_path))
    
    # 3. Train
    print("Initializing YOLO model...")
    model = get_proposed_model(model_size='s')  # Using YOLOv8s
    
    print("Starting training...")
    results = model.train(
        data=str(yaml_path),
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(OUTPUT_DIR / 'yolo_runs'),
        name='indicdoc_yolo',
        seed=SEED,
        workers=NUM_WORKERS
    )
    
    print("Training complete. Results saved to:", OUTPUT_DIR / 'yolo_runs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ann', type=str, required=True, help="Path to train annotations JSON")
    parser.add_argument('--train_img', type=str, required=True, help="Path to train images directory")
    parser.add_argument('--val_ann', type=str, required=True, help="Path to val annotations JSON")
    parser.add_argument('--val_img', type=str, required=True, help="Path to val images directory")
    parser.add_argument('--convert_annotations', action='store_true', help="Convert COCO to YOLO before training")
    args = parser.parse_args()
    main(args)
