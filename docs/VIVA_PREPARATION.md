# Viva Preparation Guide

This document contains concise, technically accurate answers to expected viva questions for the IndicDoc AI project.

**1. What problem does IndicDoc AI solve?**
It automates the extraction and understanding of complex document layouts (tables, paragraphs, headers, etc.) in multilingual Indian documents, transforming unstructured images into structured, machine-readable formats.

**2. Why was this problem selected?**
India has massive archives of physical documents across diverse languages and scripts. Manual digitization is slow and error-prone, and standard OCR fails to capture structural hierarchy (like tables and columns).

**3. Why AIKosh?**
AIKosh (IndicDLP) is the largest and most comprehensive dataset for Indian document layout parsing, offering 119K images across 12 languages and 42 layout classes, making it the academic standard for this specific problem.

**4. Why Deep Learning?**
Traditional rule-based or heuristic layout analysis methods fail on complex, degraded, or variable layouts. Deep Learning (specifically object detection) learns robust hierarchical features capable of generalizing across varied document domains and scripts.

**5. Why this architecture?**
We use Faster R-CNN (Baseline) because its Region Proposal Network (RPN) is highly accurate for precise bounding box localization. YOLOv8 (Proposed) is used because it provides real-time inference speeds and handles multi-scale features effectively via CSPDarknet.

**6. What is the input?**
A scanned document image (resized and normalized to 640x640 pixels, as a 3-channel RGB tensor).

**7. What is the output?**
A set of predicted bounding boxes, each with a specific class label (e.g., Table, Paragraph) and a confidence score.

**8. Why use transfer learning?**
Training from scratch requires millions of images. By initializing with COCO pre-trained weights, the model already understands basic visual features (edges, textures, shapes), significantly reducing training time and preventing overfitting on our dataset.

**9. What preprocessing is performed?**
Images are resized (640x640), converted to PyTorch tensors (scaling pixel values to [0,1]), and optionally normalized. We also apply data augmentation (horizontal flips, random crops, brightness adjustments) during training to improve robustness.

**10. What loss function is used?**
Faster R-CNN uses a multi-task loss: Cross-Entropy Loss for classification (what is the object?) and Smooth L1 Loss for bounding box regression (where is the object?).

**11. What optimizer is used?**
Stochastic Gradient Descent (SGD) with momentum (0.9) and weight decay (0.0005) is used for the baseline model, which helps navigate the loss landscape efficiently while preventing overfitting.

**12. What metrics are used?**
We use Intersection over Union (IoU) to measure box overlap, Precision, Recall, F1-Score, and Mean Average Precision (mAP) at various IoU thresholds (like mAP@50).

**13. What is IoU?**
Intersection over Union. It is the area of overlap between the predicted bounding box and the ground truth box, divided by the area of their union. It measures localization accuracy.

**14. What is mAP?**
Mean Average Precision. It is the area under the Precision-Recall curve, averaged across all classes. It provides a single metric to evaluate both classification and localization performance.

**15. What is the difference between Precision and Recall?**
Precision measures how many of the detected objects are correct (True Positives / All Detections). Recall measures how many of the actual ground truth objects were successfully detected (True Positives / All Ground Truths).

**16. How does the baseline differ from the proposed model?**
Faster R-CNN (baseline) is a two-stage detector, making it highly accurate but slower. YOLOv8 (proposed) is a single-stage detector, trading a small amount of accuracy for significantly faster inference speeds, making it better for real-time applications.

**17. What are the major limitations?**
The current training is limited by hardware (CPU-only), requiring us to demonstrate the pipeline on a sample dataset rather than the full 119K IndicDLP dataset. Additionally, it currently performs layout detection without integrated OCR text extraction.

**18. What are the future improvements?**
Training on the full dataset using cloud GPUs, integrating OCR (like PaddleOCR) to extract the actual text within the detected bounding boxes, and exploring multimodal architectures like LayoutLMv3.

**19. How does the project relate to SDG 9?**
It aligns with UN Sustainable Development Goal 9 (Industry, Innovation and Infrastructure) by building intelligent digital infrastructure that democratizes access to information across diverse Indian languages.

**20. Why is the project genuinely Deep Learning-based?**
It is not an API wrapper or a prompt-engineering demo. It involves downloading raw data, building a PyTorch `Dataset` loader, configuring a complex neural network architecture (Faster R-CNN), defining a training loop, computing loss via backpropagation, and extracting hierarchical features using deep convolutional layers.
