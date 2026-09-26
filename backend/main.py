import time
import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np

# Adjust imports to allow accessing the ml module from project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import DEFAULT_CLASSES, get_config
from app.streamlit_app import demo_inference

app = FastAPI(title="IndicDoc AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@app.get("/")
def read_root():
    return {"message": "Welcome to IndicDoc AI API"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), model_type: str = "yolov8", conf_threshold: float = 0.5):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # We will use demo inference for scaffolding, to ensure it runs without torch/yolo failing
        results = demo_inference(image_rgb)
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            "boxes": results["boxes"].tolist(),
            "labels": results["labels"].tolist(),
            "scores": results["scores"].tolist(),
            "inference_time": results["inference_time"],
            "is_demo": results.get("is_demo", True),
            "image_width": image.shape[1],
            "image_height": image.shape[0]
        }
        
        return {"status": "success", "results": results_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    # Return dummy/saved metrics
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    # Mocking for now
    return {
        "baseline": {
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.83,
            "map50": 0.88,
            "avg_iou": 0.76,
            "inference_time": 1.25
        },
        "proposed": {
            "precision": 0.92,
            "recall": 0.90,
            "f1": 0.91,
            "map50": 0.95,
            "avg_iou": 0.85,
            "inference_time": 0.45
        }
    }
