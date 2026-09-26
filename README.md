# IndicDoc AI

**Deep Learning-Based Multilingual Indian Document Layout Understanding**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-27%20passing-brightgreen)](#testing)

> A deep learning system for detecting and classifying document layout elements in multilingual Indian documents. Built using the IndicDLP dataset from AIKosh.

---

## Overview

IndicDoc AI is a deep learning-based system for **Document Layout Parsing (DLP)** — detecting and classifying structural elements (paragraphs, tables, headers, figures, etc.) in scanned document images. The system targets **multilingual Indian documents** across 12 languages and 12 document domains.

The project implements and compares two object detection architectures:
1. **Baseline**: Faster R-CNN with ResNet-50 FPN backbone
2. **Proposed**: YOLOv8s fine-tuned for document layout detection

Both models are evaluated on the **IndicDLP dataset** sourced from **AIKosh** (India's national AI dataset platform).

## Dataset

**IndicDLP (Indic Document Layout Parsing)**

| Property | Details |
|---|---|
| Source | [AIKosh](https://aikosh.indiaai.gov.in/home) / [HuggingFace](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset) |
| Images | 119,806 |
| Languages | 12 (Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu) |
| Domains | 12 (Novels, Textbooks, Magazines, Acts & Rules, Research Papers, Manuals, Brochures, Syllabi, Question Papers, Notices, Forms, Newspapers) |
| Annotations | COCO-style JSON |
| Classes | 42 physical and logical layout classes |

See [docs/DATASET.md](docs/DATASET.md) for detailed documentation.

## Architecture

### Baseline: Faster R-CNN
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pretrained**: COCO weights (torchvision)
- **Head**: FastRCNNPredictor for 43 classes (42 + background)
- **Optimizer**: SGD (lr=0.005, momentum=0.9)
- **Parameters**: ~41.8M | **Size**: ~160 MB

### Proposed: YOLOv8s
- **Backbone**: CSPDarknet
- **Pretrained**: COCO weights (ultralytics)
- **Training**: Fine-tuned for 42 document layout classes
- **Parameters**: ~11.2M | **Size**: ~22 MB

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Prashantj44/IndicDoc-AI.git
cd IndicDoc-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data
```bash
python -m ml.datasets.create_sample_data
```

## Usage

### Training (Sample Data)
```bash
# Run the complete training + evaluation pipeline
python scripts/run_sample_training.py
```

### Training (Custom Data)
```bash
# Train Baseline (Faster R-CNN)
python -m ml.training.train_baseline \
    --train_ann data/processed/sample/annotations.json \
    --train_img data/processed/sample/images \
    --val_ann data/processed/sample/annotations.json \
    --val_img data/processed/sample/images

# Train Proposed (YOLOv8)
python -m ml.training.train_yolo \
    --train_ann data/processed/sample/annotations.json \
    --train_img data/processed/sample/images \
    --val_ann data/processed/sample/annotations.json \
    --val_img data/processed/sample/images \
    --convert_annotations
```

### Inference
```python
from ml.inference.predict import load_model, predict_single
import torch

device = torch.device('cpu')
model = load_model('outputs/models/baseline_frcnn_best.pth', 'fasterrcnn', device)
results = predict_single(model, 'path/to/document.jpg', 'fasterrcnn', device)
print(results)
```

### Demo Application
```bash
streamlit run app/streamlit_app.py
```

### Docker
```bash
docker build -t indicdoc-ai .
docker run -p 8501:8501 indicdoc-ai
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Project Structure

```
IndicDoc-AI/
├── app/                       # Streamlit application
│   ├── assets/                # Logo and branding
│   └── streamlit_app.py       # Main application (Indic-inspired UI)
├── data/                      # Dataset directory
│   ├── raw/                   # Raw IndicDLP data
│   └── processed/sample/      # Synthetic sample data
├── ml/                        # Core ML package
│   ├── config.py              # Project configuration
│   ├── datasets/              # Dataset loading & sample generation
│   ├── preprocessing/         # Image preprocessing & augmentation
│   ├── models/                # Model definitions (FRCNN + YOLOv8)
│   ├── training/              # Training scripts
│   ├── evaluation/            # Metrics & visualization
│   └── inference/             # Inference pipeline
├── outputs/                   # Training outputs
│   ├── models/                # Saved model weights
│   ├── predictions/           # Prediction images & error analysis
│   ├── metrics/               # Evaluation metrics (JSON)
│   └── plots/                 # Training curves & charts
├── docs/                      # Documentation
│   ├── DATASET.md
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   ├── EVALUATION.md
│   ├── RESULTS.md
│   ├── LIMITATIONS.md
│   ├── ACADEMIC_REPORT.md
│   └── PROGRESS.md
├── tests/                     # Test suite (27 tests)
├── scripts/                   # Utility scripts
│   └── run_sample_training.py # Complete training + evaluation pipeline
├── notebooks/                 # Analysis notebooks (Python scripts)
├── Dockerfile                 # Docker configuration
├── requirements.txt
├── .gitignore
└── README.md
```

## Evaluation

### Training Results (Sample Data — 30 images, 5 epochs)

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 5.2296 | 3.7781 |
| 2 | 3.1862 | 2.5012 |
| 3 | 2.3658 | 2.2356 |
| 4 | 2.1096 | 2.2160 |
| 5 | 2.0522 | 2.1855 |

### Model Comparison

| Metric | Faster R-CNN (Baseline) | YOLOv8 (Proposed) |
|--------|------------------------|-------------------|
| Precision | 0.0* | Not trained yet |
| Recall | 0.0* | Not trained yet |
| F1-Score | 0.0* | Not trained yet |
| mAP@50 | 0.0* | Not trained yet |
| Inference Time | 1.61s/image (CPU) | Not trained yet |
| Model Size | ~160 MB | ~22 MB |

> *Zero metrics are expected on 30 synthetic images with 42 classes. The model has fewer than 1 image per class. This demonstrates pipeline correctness. See [docs/RESULTS.md](docs/RESULTS.md) for details.

### Testing

```
27 passed in 26.25s
```

All tests pass:
- Configuration, preprocessing, transforms
- Dataset loading, sample data creation
- Metrics (IoU, AP, confusion matrix)
- Model creation and forward pass
- Visualization functions
- COCO-to-YOLO conversion
- End-to-end pipeline

## Demo Application

The Streamlit application features an **Indic-inspired design system**:
- Deep Indigo + Antique Gold + Warm Ivory color palette
- 5 tabs: Dashboard, Analyze, Model Insights, Evaluation, About
- Real-time document layout detection with bounding box visualization
- Model comparison dashboard with charts
- Download annotated results
- Demo mode when no trained model is available

```bash
streamlit run app/streamlit_app.py
```

## SDG Alignment

**SDG 9: Industry, Innovation and Infrastructure**
- Enables AI-based document digitization for Indian government, educational, and legal documents
- Supports multilingual information processing across 12 Indian languages
- Contributes to intelligent document infrastructure

## Limitations

1. Training demonstrated on synthetic sample data (full dataset requires GPU)
2. Transfer learning from COCO (not trained from scratch)
3. OCR not integrated (layout detection only)
4. Per-language evaluation pending full dataset

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for full details.

## Future Scope

1. Training on the full IndicDLP dataset (119K images) with GPU
2. Integration with OCR engines (PaddleOCR/Tesseract) for text extraction
3. LayoutLMv3 for joint layout + text understanding
4. Document structure hierarchy reconstruction
5. ONNX/TFLite conversion for mobile deployment
6. Per-language performance analysis

## References

1. IndicDLP Dataset — AIKosh / AI4Bharat
2. Ren, S., et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS 2015.
3. Jocher, G., et al. "Ultralytics YOLOv8." 2023.
4. Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
5. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
6. AIKosh — IndiaAI Mission, MeitY, Government of India.

---

**B.E. Artificial Intelligence & Machine Learning — Deep Learning Mini-Project**

*SDG 9: Industry, Innovation and Infrastructure*
