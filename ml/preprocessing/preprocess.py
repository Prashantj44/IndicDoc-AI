import cv2
import numpy as np
from typing import Tuple

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def denormalize_image(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img = image.copy()
    if img.ndim == 3 and img.shape[0] == 3:  # CHW format
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
    elif img.ndim == 3 and img.shape[2] == 3:  # HWC format
        for c in range(3):
            img[:,:,c] = img[:,:,c] * std[c] + mean[c]
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew a document image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_document(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for documents."""
    img = deskew_image(image)
    img = enhance_contrast(img)
    return img
