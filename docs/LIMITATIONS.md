# Limitations

## Current Limitations

### 1. Hardware Constraints
- Training and inference performed on CPU (no GPU available)
- Full IndicDLP dataset (119K images) requires significant GPU memory
- Current training demonstrated on 30 synthetic sample images

### 2. Dataset Limitations
- Demo uses synthetic sample data, not the full IndicDLP dataset
- Full dataset requires download from HuggingFace (~119,806 images)
- Per-language evaluation requires language metadata in annotations

### 3. Training Scope
- Models are fine-tuned (transfer learning) rather than trained from scratch
- Limited hyperparameter search due to hardware constraints
- Only 5 training epochs on sample data (production would need 50-100+)

### 4. Model Limitations
- Baseline (Faster R-CNN) is slower for real-time applications
- YOLOv8 may struggle with very small document elements
- Both models may confuse visually similar classes (e.g., Header vs. Title)

### 5. Functionality
- OCR is not integrated in the current version (layout detection only)
- No document hierarchy reconstruction (flat detection only)
- Single-page analysis only (no multi-page document support)

### 6. Evaluation
- Evaluation metrics on sample data show pipeline correctness, not production performance
- Real performance requires training on the full IndicDLP dataset
- Confusion matrix generated on synthetic data for demonstration

## Mitigation Strategies

1. **GPU Training**: Use Google Colab or cloud GPUs for full dataset training
2. **Full Dataset**: Download IndicDLP from HuggingFace for production training
3. **Extended Training**: Increase epochs to 50-100 with learning rate scheduling
4. **Hyperparameter Tuning**: Grid search or Bayesian optimization for lr, batch size, augmentation
5. **OCR Integration**: Add PaddleOCR or Tesseract for text extraction
