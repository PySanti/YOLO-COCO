from torch import nn
from utils.MACROS import *
import torch
import torch.nn as nn
from utils.ConvBlock import ConvBlock


class YOLOV1Backbone(nn.Module):
    def __init__(self):
        super(YOLOV1Backbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)


class YOLOV1Head(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOV1Head, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.backbone = YOLOV1Backbone()
        self.head = YOLOV1Head(GRID_SIZE, NUM_CLASSES, 1)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
