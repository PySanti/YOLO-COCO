import torch
def yolo_iou(boxes1, boxes2, eps=1e-6):
    """
    IoU for boxes in (x_center, y_center, w, h) format.
    boxes1, boxes2: (..., 4)
    """
    # Ensure positive sizes
    w1 = boxes1[..., 2].clamp(min=0)
    h1 = boxes1[..., 3].clamp(min=0)
    w2 = boxes2[..., 2].clamp(min=0)
    h2 = boxes2[..., 3].clamp(min=0)

    x1, y1 = boxes1[..., 0], boxes1[..., 1]
    x2, y2 = boxes2[..., 0], boxes2[..., 1]

    # Convert to corners
    b1_x1 = x1 - w1 / 2
    b1_y1 = y1 - h1 / 2
    b1_x2 = x1 + w1 / 2
    b1_y2 = y1 + h1 / 2

    b2_x1 = x2 - w2 / 2
    b2_y1 = y2 - h2 / 2
    b2_x2 = x2 + w2 / 2
    b2_y2 = y2 + h2 / 2

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)

    union = area1 + area2 - inter_area
    return inter_area / (union + eps)


