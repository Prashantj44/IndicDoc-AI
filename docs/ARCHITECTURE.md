# Architecture

## Pipeline Overview

```
Document Image
      |
      v
Preprocessing (Resize 640x640, Normalize, CLAHE, Deskew)
      |
      v
Augmentation (HFlip, ColorJitter, RandomCrop - training only)
      |
      v
Deep Learning Model (Faster R-CNN / YOLOv8)
      |
      v
Non-Maximum Suppression (NMS)
      |
      v
Bounding Boxes + Class Labels + Confidence Scores
      |
      v
Visualization & Evaluation
```

## Model Architectures

### Baseline: Faster R-CNN with ResNet-50 FPN

| Property | Details |
|---|---|
| **Architecture** | Faster R-CNN |
| **Backbone** | ResNet-50 with Feature Pyramid Network (FPN) |
| **Detector Type** | Two-stage (RPN + Fast R-CNN) |
| **Pretrained Weights** | COCO (torchvision `fasterrcnn_resnet50_fpn`) |
| **Detection Head** | FastRCNNPredictor (modified for 43 classes) |
| **Output Classes** | 42 IndicDLP classes + 1 background = 43 |
| **Parameters** | ~41.8 million |
| **Model Size** | ~160 MB |

**Training Configuration:**
- Optimizer: SGD (lr=0.005, momentum=0.9, weight_decay=5e-4)
- Scheduler: StepLR (step_size=3, gamma=0.1)
- Gradient Clipping: max_norm=10.0
- Input Size: 640 x 640

### Proposed: YOLOv8s

| Property | Details |
|---|---|
| **Architecture** | YOLOv8s (small variant) |
| **Backbone** | CSPDarknet |
| **Detector Type** | Single-stage (anchor-free) |
| **Pretrained Weights** | COCO (ultralytics) |
| **Output Classes** | 42 IndicDLP classes |
| **Parameters** | ~11.2 million |
| **Model Size** | ~22 MB |

**Training Configuration:**
- Optimizer: SGD (ultralytics auto-configured)
- Scheduler: Cosine annealing
- Augmentation: Mosaic, MixUp, HSV, Horizontal Flip
- Input Size: 640 x 640

### Architecture Comparison

| Feature | Faster R-CNN | YOLOv8s |
|---|---|---|
| Detection Approach | Two-stage | Single-stage |
| Region Proposal | RPN (Region Proposal Network) | Anchor-free |
| Feature Extraction | ResNet-50 + FPN | CSPDarknet |
| Speed | Slower (two-pass) | Faster (single-pass) |
| Accuracy (general) | Higher precision | Competitive |
| Model Size | ~160 MB | ~22 MB |
| Inference Speed | Slower | ~2.5x faster |

### Transfer Learning Strategy

Both models use **transfer learning** from COCO-pretrained weights:

1. Load pretrained weights (trained on 80 COCO classes)
2. Replace the classification head for 42 IndicDLP classes
3. Fine-tune all layers with a reduced learning rate
4. Benefits: Faster convergence, better generalization on limited data

### Technical Justification

**Why Faster R-CNN as baseline?**
- Gold standard for object detection research
- Strong performance on small/overlapping objects
- Well-studied architecture with extensive benchmarks
- Torchvision provides reliable implementation

**Why YOLOv8 as proposed?**
- State-of-the-art single-stage detector (2023)
- Significantly faster inference for real-time applications
- Smaller model size suitable for deployment
- Built-in augmentation pipeline (Mosaic, MixUp)
- Anchor-free design handles varying object sizes well
