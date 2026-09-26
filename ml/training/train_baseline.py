import os
import argparse
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.config import *
from ml.datasets.coco_dataset import COCODataset, collate_fn
from ml.preprocessing.transforms import get_train_transforms, get_val_transforms
from ml.models.baseline_frcnn import get_baseline_model

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize datasets
    train_dataset = COCODataset(
        annotation_file=args.train_ann,
        img_dir=args.train_img,
        transforms=get_train_transforms(IMG_SIZE)
    )
    
    val_dataset = COCODataset(
        annotation_file=args.val_ann,
        img_dir=args.val_img,
        transforms=get_val_transforms(IMG_SIZE)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Model (classes + 1 for background)
    model = get_baseline_model(num_classes=NUM_CLASSES + 1, pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    scaler = torch.amp.GradScaler('cuda') if MIXED_PRECISION and device.type == 'cuda' else None

    if MLFLOW_AVAILABLE and args.use_mlflow:
        mlflow.start_run()
        mlflow.log_params({"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS})

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

            train_loss += losses.item()
            progress_bar.set_postfix({'loss': losses.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        if MLFLOW_AVAILABLE and args.use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        torch.save(model.state_dict(), MODEL_DIR / f'baseline_frcnn_ep{epoch+1}.pth')

    if MLFLOW_AVAILABLE and args.use_mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ann', type=str, required=True, help="Path to train annotations JSON")
    parser.add_argument('--train_img', type=str, required=True, help="Path to train images directory")
    parser.add_argument('--val_ann', type=str, required=True, help="Path to val annotations JSON")
    parser.add_argument('--val_img', type=str, required=True, help="Path to val images directory")
    parser.add_argument('--use_mlflow', action='store_true', help="Enable MLflow logging")
    args = parser.parse_args()
    main(args)
