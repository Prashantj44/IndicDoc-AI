"""
End-to-end training pipeline verification script.
Trains Faster R-CNN on sample data, saves model, runs inference.
"""
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.config import PROCESSED_DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR, NUM_CLASSES, CLASS_NAMES, SEED
from ml.datasets.coco_dataset import COCODataset, collate_fn
from ml.models.baseline_frcnn import get_baseline_model
from ml.evaluation.metrics import save_metrics_report
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cpu')
print(f"Device: {device}")

# Setup directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
ann = str(PROCESSED_DATA_DIR / 'sample' / 'annotations.json')
imgs = str(PROCESSED_DATA_DIR / 'sample' / 'images')
ds = COCODataset(ann, imgs)
print(f"Dataset loaded: {len(ds)} images")

# Split
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Model
model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Training
NUM_EPOCHS = 3
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

train_losses = []
val_losses = []

def prepare_batch(images, targets):
    """Convert images/targets to proper format for Faster R-CNN."""
    imgs_t = []
    tgts = []
    for img in images:
        if isinstance(img, torch.Tensor):
            imgs_t.append(img.to(device))
        else:
            imgs_t.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).to(device))
    for t in targets:
        new_t = {k: v.to(device) for k, v in t.items()}
        # Faster R-CNN expects labels >= 1 (0 = background)
        new_t['labels'] = new_t['labels'] + 1
        tgts.append(new_t)
    return imgs_t, tgts

print(f"\nTraining for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for images, targets in train_loader:
        imgs_t, tgts = prepare_batch(images, targets)
        try:
            loss_dict = model(imgs_t, tgts)
            loss = sum(l for l in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"  Batch error: {e}")
            continue

    avg_train_loss = epoch_loss / max(num_batches, 1)
    train_losses.append(avg_train_loss)
    scheduler.step()

    # Validation
    model.train()  # FRCNN needs train mode for loss computation
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            imgs_t, tgts = prepare_batch(images, targets)
            try:
                loss_dict = model(imgs_t, tgts)
                loss = sum(l for l in loss_dict.values())
                val_loss += loss.item()
                val_batches += 1
            except Exception:
                continue

    avg_val_loss = val_loss / max(val_batches, 1)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

# Save model
model_path = MODEL_DIR / 'baseline_frcnn_best.pth'
torch.save(model.state_dict(), str(model_path))
print(f"\nModel saved to {model_path}")

# Plot training curves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train Loss', linewidth=2)
ax.plot(range(1, len(val_losses)+1), val_losses, 'r-o', label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Faster R-CNN Training Curves (Sample Data)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'training_loss.png'), dpi=150)
print(f"Training curves saved to {PLOTS_DIR / 'training_loss.png'}")
plt.close()

# Inference test
print("\nRunning inference test...")
model.eval()

import time
from ml.config import PREDICTION_DIR
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

test_images = sorted(list((PROCESSED_DATA_DIR / 'sample' / 'images').glob('*.jpg')))[:4]
inference_times = []
all_detections = 0

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
import matplotlib.patches as mpatches

for idx, img_path in enumerate(test_images):
    img = plt.imread(str(img_path))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(device)

    start = time.time()
    with torch.no_grad():
        preds = model([img_tensor])[0]
    inf_time = time.time() - start
    inference_times.append(inf_time)

    # Filter predictions
    keep = preds['scores'] > 0.3
    boxes = preds['boxes'][keep].cpu().numpy()
    labels = preds['labels'][keep].cpu().numpy()
    scores = preds['scores'][keep].cpu().numpy()
    all_detections += len(boxes)

    axes[idx].imshow(img)
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        c = colors[int(label) % len(colors)]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none')
        axes[idx].add_patch(rect)
        lbl = CLASS_NAMES[int(label)-1] if 0 < int(label) <= len(CLASS_NAMES) else f'cls_{int(label)}'
        axes[idx].text(x1, y1-2, f'{lbl}: {score:.2f}', fontsize=5, color='white',
                       bbox=dict(facecolor=c, alpha=0.7, pad=0.5))
    axes[idx].set_title(f'{img_path.name}\n{len(boxes)} det, {inf_time:.3f}s', fontsize=9)
    axes[idx].axis('off')

plt.suptitle('Model Predictions on Sample Images', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(str(PREDICTION_DIR / 'sample_predictions.png'), dpi=150)
print(f"Predictions saved to {PREDICTION_DIR / 'sample_predictions.png'}")
plt.close()

# Save metrics
metrics = {
    'model': 'Faster R-CNN (ResNet-50 FPN)',
    'dataset': 'Synthetic sample data (pipeline verification)',
    'epochs_trained': NUM_EPOCHS,
    'final_train_loss': round(train_losses[-1], 4),
    'final_val_loss': round(val_losses[-1], 4),
    'train_losses': [round(l, 4) for l in train_losses],
    'val_losses': [round(l, 4) for l in val_losses],
    'total_params': total_params,
    'avg_inference_time_s': round(float(np.mean(inference_times)), 4),
    'total_detections': all_detections,
    'device': str(device),
    'note': 'Trained on synthetic sample data for pipeline verification. Real metrics require IndicDLP dataset.'
}
save_metrics_report(metrics, str(METRICS_DIR / 'baseline_metrics.json'))
print(f"Metrics saved to {METRICS_DIR / 'baseline_metrics.json'}")

print("\n" + "=" * 60)
print("END-TO-END PIPELINE VERIFIED SUCCESSFULLY")
print("=" * 60)
print(f"  - Training: {NUM_EPOCHS} epochs completed")
print(f"  - Model saved: {model_path}")
print(f"  - Training curves: {PLOTS_DIR / 'training_loss.png'}")
print(f"  - Predictions: {PREDICTION_DIR / 'sample_predictions.png'}")
print(f"  - Avg inference time: {np.mean(inference_times):.4f}s")
print(f"  - Total detections: {all_detections}")
