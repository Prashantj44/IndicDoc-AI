# Methodology

## 1. Problem Formulation

Document Layout Parsing (DLP) is formulated as an **object detection** problem:
- **Input**: Scanned document image (RGB)
- **Output**: Bounding boxes + class labels + confidence scores for each detected document element
- **Classes**: 42 physical and logical layout elements

## 2. Data Pipeline

### 2.1 Data Loading
- Parse COCO-format JSON annotations
- Custom PyTorch `Dataset` class (`COCODataset`)
- Load images using PIL, convert to RGB
- Map annotations to image IDs

### 2.2 Preprocessing
| Step | Method | Purpose |
|---|---|---|
| Resize | Bilinear interpolation to 640x640 | Standard input size |
| Normalization | Scale to [0, 1] / ImageNet stats | Model compatibility |
| CLAHE | Contrast Limited Adaptive Histogram Equalization | Enhance document readability |
| Deskewing | MinAreaRect angle estimation + affine transform | Correct document rotation |

### 2.3 Data Augmentation (Training Only)
| Augmentation | Parameter | Purpose |
|---|---|---|
| Horizontal Flip | p=0.5 | Spatial invariance |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 | Color robustness |
| Random Crop | min_visibility=0.2 | Spatial diversity |

### 2.4 Train/Val/Test Split
- Training: 70%
- Validation: 15%
- Testing: 15%
- Reproducible split with seed=42

## 3. Modeling Strategy

### 3.1 Two-Model Comparison
| | Baseline | Proposed |
|---|---|---|
| Model | Faster R-CNN | YOLOv8s |
| Backbone | ResNet-50 + FPN | CSPDarknet |
| Type | Two-stage | Single-stage |
| Pretrained | COCO (torchvision) | COCO (ultralytics) |

### 3.2 Transfer Learning
1. Load COCO-pretrained weights (80 classes)
2. Replace detection head for 42 IndicDLP classes (+1 background for Faster R-CNN)
3. Fine-tune all layers

### 3.3 Training Protocol
- **Optimizer**: SGD (Faster R-CNN) / SGD auto (YOLOv8)
- **Learning Rate**: 0.005 with step decay
- **Gradient Clipping**: max_norm=10.0
- **Mixed Precision**: Enabled when CUDA available
- **Early Stopping**: Monitor validation loss

## 4. Evaluation Protocol

### 4.1 Metrics
| Metric | Description |
|---|---|
| **IoU** | Intersection over Union between predicted and ground truth boxes |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | 2 * P * R / (P + R) |
| **mAP@50** | Mean Average Precision at IoU >= 0.50 |
| **mAP@50:95** | Mean AP over IoU thresholds 0.50 to 0.95 (step 0.05) |
| **Inference Time** | Average prediction time per image |

### 4.2 Evaluation Procedure
1. Model inference on test set (confidence threshold = 0.5)
2. Non-Maximum Suppression (NMS) for overlapping detections
3. IoU matching between predictions and ground truth (threshold = 0.5)
4. Per-class precision, recall, and AP computation
5. Aggregate metrics across all classes

## 5. Error Analysis

Systematic analysis of model failures:
- **Missed detections**: Small elements, low-contrast regions
- **False positives**: Background regions, decorative elements
- **Misclassifications**: Similar classes (Header vs. Title)
- **Localization errors**: Imprecise bounding boxes, merged elements

## 6. Reproducibility

- Fixed random seed (42) across all experiments
- Deterministic data splits
- Configuration stored in `ml/config.py`
- Training logs saved in `outputs/metrics/`
