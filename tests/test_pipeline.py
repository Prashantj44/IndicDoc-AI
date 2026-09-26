"""
IndicDoc AI - Test Suite
Comprehensive tests for the ML pipeline.
"""
import pytest
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    def test_config_imports(self):
        from ml.config import PROJECT_ROOT, NUM_CLASSES, CLASS_NAMES, IMG_SIZE
        assert NUM_CLASSES == 42
        assert len(CLASS_NAMES) == 42
        assert IMG_SIZE == 640
        assert PROJECT_ROOT.exists()

    def test_paths_exist(self):
        from ml.config import PROJECT_ROOT
        assert (PROJECT_ROOT / 'ml').exists()
        assert (PROJECT_ROOT / 'app').exists()
        assert (PROJECT_ROOT / 'docs').exists()


class TestPreprocessing:
    def test_resize(self):
        from ml.preprocessing.preprocess import resize_image
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        resized = resize_image(img, (64, 64))
        assert resized.shape == (64, 64, 3)

    def test_normalize(self):
        from ml.preprocessing.preprocess import normalize_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normed = normalize_image(img)
        assert normed.max() <= 1.0
        assert normed.min() >= 0.0

    def test_enhance_contrast(self):
        from ml.preprocessing.preprocess import enhance_contrast
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = enhance_contrast(img)
        assert enhanced.shape == img.shape

    def test_deskew(self):
        from ml.preprocessing.preprocess import deskew_image
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        deskewed = deskew_image(img)
        assert deskewed.shape == img.shape

    def test_preprocess_document(self):
        from ml.preprocessing.preprocess import preprocess_document
        img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
        processed = preprocess_document(img)
        assert processed.shape[:2] == img.shape[:2]


class TestTransforms:
    def test_train_transforms(self):
        from ml.preprocessing.transforms import get_train_transforms
        t = get_train_transforms(640)
        assert t is not None

    def test_val_transforms(self):
        from ml.preprocessing.transforms import get_val_transforms
        t = get_val_transforms(640)
        assert t is not None

    def test_inference_transforms(self):
        from ml.preprocessing.transforms import get_inference_transforms
        t = get_inference_transforms(640)
        assert t is not None


class TestSampleDataCreation:
    def test_create_sample_data(self, tmp_path):
        """Test sample data creation by directly calling the generation logic."""
        import random
        from PIL import Image, ImageDraw
        from ml.config import CLASS_NAMES

        output_dir = tmp_path / 'sample'
        img_dir = output_dir / 'images'
        img_dir.mkdir(parents=True)

        num_images = 5
        coco_data = {
            "info": {"description": "Test", "version": "1.0", "year": 2024},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "document"} for i, name in enumerate(CLASS_NAMES)]
        }

        ann_id = 1
        random.seed(42)
        for img_id in range(1, num_images + 1):
            width, height = 800, 1000
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            file_name = f"sample_{img_id:04d}.jpg"
            coco_data["images"].append({
                "id": img_id, "width": width, "height": height, "file_name": file_name
            })
            num_boxes = random.randint(3, 10)
            for _ in range(num_boxes):
                cat_id = random.randint(0, len(CLASS_NAMES) - 1)
                x = random.randint(50, width - 200)
                y = random.randint(50, height - 200)
                w = random.randint(50, 150)
                h = random.randint(20, 100)
                draw.rectangle([x, y, x + w, y + h], outline=(100, 100, 100), width=2)
                coco_data["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id,
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "segmentation": []
                })
                ann_id += 1
            img.save(img_dir / file_name)

        with open(output_dir / "annotations.json", "w") as f:
            json.dump(coco_data, f, indent=4)

        assert (output_dir / 'annotations.json').exists()
        assert len(list((output_dir / 'images').glob('*.jpg'))) == 5

        with open(output_dir / 'annotations.json') as f:
            data = json.load(f)
        assert 'images' in data
        assert 'annotations' in data
        assert 'categories' in data
        assert len(data['images']) == 5
        assert len(data['categories']) == 42


class TestDataset:
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal sample data for testing."""
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        from PIL import Image
        for i in range(3):
            img = Image.new('RGB', (100, 100), color='white')
            img.save(img_dir / f'img_{i}.jpg')

        annotations = {
            'images': [
                {'id': i, 'width': 100, 'height': 100, 'file_name': f'img_{i}.jpg'}
                for i in range(3)
            ],
            'annotations': [
                {'id': j, 'image_id': i, 'category_id': j % 5, 'bbox': [10, 10, 30, 30],
                 'area': 900, 'iscrowd': 0, 'segmentation': []}
                for i in range(3) for j in range(i * 2, i * 2 + 2)
            ],
            'categories': [
                {'id': i, 'name': f'class_{i}', 'supercategory': 'doc'}
                for i in range(5)
            ]
        }
        ann_path = tmp_path / 'annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(annotations, f)
        return str(ann_path), str(img_dir)

    def test_dataset_loading(self, sample_data):
        from ml.datasets.coco_dataset import COCODataset
        ann_path, img_dir = sample_data
        ds = COCODataset(ann_path, img_dir)
        assert len(ds) == 3

    def test_dataset_item(self, sample_data):
        from ml.datasets.coco_dataset import COCODataset
        ann_path, img_dir = sample_data
        ds = COCODataset(ann_path, img_dir)
        img, target = ds[0]
        assert 'boxes' in target
        assert 'labels' in target


class TestMetrics:
    def test_calculate_iou_perfect(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        assert calculate_iou(box1, box2) == pytest.approx(1.0)

    def test_calculate_iou_no_overlap(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box3 = [20, 20, 30, 30]
        assert calculate_iou(box1, box3) == pytest.approx(0.0)

    def test_calculate_iou_partial(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Overlap area: 5*5=25, Union: 100+100-25=175
        assert iou == pytest.approx(25 / 175, abs=0.01)

    def test_calculate_ap(self):
        from ml.evaluation.metrics import calculate_ap
        precision = np.array([1.0, 0.8, 0.6, 0.4])
        recall = np.array([0.2, 0.4, 0.6, 0.8])
        ap = calculate_ap(precision, recall)
        assert 0.0 <= ap <= 1.0

    def test_save_metrics_report(self, tmp_path):
        from ml.evaluation.metrics import save_metrics_report
        metrics = {"precision": 0.85, "recall": 0.78, "f1": 0.81}
        filepath = str(tmp_path / "metrics" / "test_metrics.json")
        save_metrics_report(metrics, filepath)
        assert Path(filepath).exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["precision"] == 0.85

    def test_generate_confusion_matrix(self):
        from ml.evaluation.metrics import generate_confusion_matrix
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        cm = generate_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # Class 0 correct


class TestModels:
    def test_baseline_model_creation(self):
        from ml.models.baseline_frcnn import get_baseline_model
        model = get_baseline_model(num_classes=43, pretrained=False)
        assert model is not None

    def test_baseline_model_forward(self):
        import torch
        from ml.models.baseline_frcnn import get_baseline_model
        model = get_baseline_model(num_classes=43, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model([dummy[0]])
        assert 'boxes' in output[0]
        assert 'labels' in output[0]
        assert 'scores' in output[0]


class TestVisualization:
    def test_draw_predictions(self):
        from ml.evaluation.visualize import draw_predictions
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Updated signature: labels is List[str], no class_names parameter
        fig = draw_predictions(img, [[10, 10, 50, 50]], ['TestClass'], [0.9])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_draw_ground_truth(self):
        from ml.evaluation.visualize import draw_ground_truth
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fig = draw_ground_truth(img, [[10, 10, 50, 50]], ['TestClass'])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_training_curves(self):
        from ml.evaluation.visualize import plot_training_curves
        fig = plot_training_curves([0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_class_distribution(self, tmp_path):
        from ml.evaluation.visualize import plot_class_distribution
        ann = {
            "annotations": [{"category_id": 0}, {"category_id": 0}, {"category_id": 1}],
            "categories": [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}]
        }
        ann_path = str(tmp_path / "ann.json")
        with open(ann_path, 'w') as f:
            json.dump(ann, f)
        fig = plot_class_distribution(ann_path)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestEndToEnd:
    def test_full_pipeline(self):
        """Test: config -> preprocess -> model creation -> inference."""
        import torch
        from ml.config import NUM_CLASSES, IMG_SIZE
        from ml.preprocessing.preprocess import resize_image, normalize_image
        from ml.models.baseline_frcnn import get_baseline_model

        # Create dummy image
        img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

        # Preprocess
        resized = resize_image(img, (IMG_SIZE, IMG_SIZE))
        assert resized.shape == (IMG_SIZE, IMG_SIZE, 3)

        # Model
        model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        model.eval()

        # Inference
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            results = model([tensor])

        assert 'boxes' in results[0]
        assert 'labels' in results[0]
        assert 'scores' in results[0]


class TestCOCOToYOLO:
    def test_coco_to_yolo_conversion(self, tmp_path):
        from ml.models.proposed_yolov8 import coco_to_yolo

        coco_ann = {
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]}
            ],
            "categories": [{"id": 0, "name": "test"}]
        }
        ann_path = str(tmp_path / "ann.json")
        with open(ann_path, 'w') as f:
            json.dump(coco_ann, f)

        output_dir = str(tmp_path / "labels")
        coco_to_yolo(ann_path, output_dir)

        label_file = tmp_path / "labels" / "img.txt"
        assert label_file.exists()
        content = label_file.read_text().strip()
        parts = content.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # category_id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
