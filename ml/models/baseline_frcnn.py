import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

def get_baseline_model(num_classes: int, pretrained: bool = True):
    """
    Returns a Faster R-CNN model modified for the specified number of classes.
    Note: num_classes should include the background class (e.g., 42 IndicDLP classes + 1 bg = 43).
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Optionally, configure custom anchors if needed for document objects
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    # model.rpn.anchor_generator = anchor_generator
    
    return model
