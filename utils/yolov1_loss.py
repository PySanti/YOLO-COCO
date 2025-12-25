import torch

def yolov1_loss(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    # Unpack predictions and targets
    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]
    
    # Localization Loss

    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    # Confidence Loss
    obj_loss = torch.sum((pred_conf - target_conf) ** 2)
    noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

    # Classification Loss
    class_loss = torch.sum((pred_classes - target_classes) ** 2)

    # Total Loss
    total_loss = box_loss + obj_loss + noobj_loss + class_loss
    return total_loss
