import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
MODEL_DIR = PROJECT_ROOT / 'outputs' / 'models'
PREDICTION_DIR = OUTPUT_DIR / 'predictions'
METRICS_DIR = OUTPUT_DIR / 'metrics'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Dataset config
NUM_CLASSES = 42  # IndicDLP classes
CLASS_NAMES = [
    'Paragraph', 'Image', 'Table', 'Header', 'Footer',
    'Title', 'Caption', 'Page Number', 'Footnote', 'List',
    'Figure', 'Equation', 'Logo', 'Stamp', 'Signature',
    'Handwriting', 'Chart', 'Map', 'Separator', 'Advertisement',
    'Watermark', 'Background', 'Margin Note', 'Column',
    'Section Header', 'Sub Header', 'Abstract', 'Author',
    'Affiliation', 'Date', 'Reference', 'Acknowledgment',
    'Appendix', 'Table of Contents', 'Index', 'Glossary',
    'Bibliography', 'Preface', 'Dedication', 'Colophon',
    'Running Header', 'Running Footer'
]

# Training config
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 640
SEED = 42
NUM_WORKERS = 2
DEVICE = 'cuda'  # Will be auto-detected
MIXED_PRECISION = True
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model config
BASELINE_MODEL = 'fasterrcnn'  
PROPOSED_MODEL = 'yolov8'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
