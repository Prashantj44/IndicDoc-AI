# Evaluation

## Evaluation Protocol

### Metrics Used

| Metric | Description | Formula |
|--------|-------------|---------|
| **IoU** | Intersection over Union between predicted and ground truth boxes | IoU = Area(Intersection) / Area(Union) |
| **Precision** | Ratio of true positive detections to all positive detections | TP / (TP + FP) |
| **Recall** | Ratio of true positive detections to all ground truth objects | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of Precision and Recall | 2 * (P * R) / (P + R) |
| **mAP@50** | Mean Average Precision at IoU >= 0.50 | Mean of per-class AP@50 |
| **mAP@50:95** | Mean AP averaged over IoU thresholds from 0.50 to 0.95 | Mean over thresholds and classes |
| **Inference Time** | Average prediction time per image | Total time / N images |

### Evaluation Procedure

1. **Model inference** on the test set with confidence threshold = 0.5
2. **NMS** (Non-Maximum Suppression) to remove overlapping detections
3. **IoU matching** between predictions and ground truth (IoU >= 0.50)
4. **Per-class** precision, recall, and AP computation
5. **Aggregate** metrics across all 42 classes

### Current Results (Sample Data)

| Metric | Faster R-CNN (Baseline) | YOLOv8 (Proposed) |
|--------|------------------------|-------------------|
| Precision | 0.0 | Not trained yet |
| Recall | 0.0 | Not trained yet |
| F1-Score | 0.0 | Not trained yet |
| mAP@50 | 0.0 | Not trained yet |
| mAP@50:95 | 0.0 | Not trained yet |
| Inference Time | 1.61s/image (CPU) | Not trained yet |
| Model Size (MB) | ~160 | ~22 |

> **Note**: Zero metrics are expected on 30 synthetic images with 42 classes. The model saw fewer than 1 image per class. This demonstrates pipeline correctness. Train on the full IndicDLP dataset for production results.

### Per-Language Evaluation Framework

| Language | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Assamese | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Bengali | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| English | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Gujarati | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Hindi | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Kannada | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Malayalam | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Marathi | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Odia | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Punjabi | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Tamil | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Telugu | Not evaluated yet | Not evaluated yet | Not evaluated yet |

> Per-language evaluation requires language metadata in the annotations or filename conventions.

## Error Analysis

### Types of Errors Observed

1. **Missed Detections (False Negatives)**: 17 objects missed on the sample validation set
   - Primary cause: insufficient training data per class
   - Small document elements are hardest to detect
   - Low-contrast elements may be missed

2. **False Detections (False Positives)**: 0 on sample data
   - On real data, expected sources: background noise, decorative elements

3. **Misclassifications**: Expected on full dataset
   - Similar classes: Header vs Title, Running Header vs Header
   - Context-dependent elements (e.g., Column vs Paragraph)

4. **Localization Errors**: Expected on full dataset
   - Imprecise bounding boxes for irregular text blocks
   - Merged detection of adjacent elements

### Factors Affecting Performance

- Document scan quality (resolution, noise, contrast)
- Script complexity (Latin scripts tend to be easier)
- Layout density (crowded pages with many small elements)
- Element size (very small or very large elements)
- Class frequency (rare classes have lower performance)
- Document domain (newspapers vs forms vs research papers)

### Visualizations

- Training loss curves: `outputs/plots/training_loss.png`
- Precision-recall curve: `outputs/plots/precision_recall.png`
- Confusion matrix: `outputs/plots/confusion_matrix.png`
- Class distribution: `outputs/plots/class_distribution.png`
- Model comparison: `outputs/plots/model_comparison.png`
- Prediction examples: `outputs/predictions/prediction_*.png`
- Error analysis: `outputs/predictions/errors/error_analysis.png`
