from torch import nn
import torch
from utils.MACROS import *
import torch.nn as nn
from utils.ConvBlock import ConvBlock
from utils.SEBlock import SEBlock
from utils.ResBlock import ResBlock
import torch.nn.functional as F


class YOLOV1Backbone(nn.Module):
    def __init__(self):
        super(YOLOV1Backbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, padding=3, stride=2),
            ConvBlock(32, 64, kernel_size=5, padding=2, stride=2),
            ResBlock(64, 96),
            SEBlock(96),
            ResBlock(96, 128, downsample=True),
            SEBlock(128),
            ResBlock(128, 160, downsample=True),
            SEBlock(160),
            ResBlock(160, 192, downsample=True),
            SEBlock(192),
            ResBlock(192, 192, downsample=True),
            SEBlock(192),
            ResBlock(192, 256, downsample=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d(GRID_SIZE)
        )

    def forward(self, x):
        return self.layers(x)


class YOLOV1Head(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOV1Head, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(256, num_anchors * (5 + num_classes), kernel_size=1)

    def _activate_output(self, out):
        anchors = []
        for a in range(self.num_anchors):
            ll = a*(5+self.num_classes)
            rl = (a+1)*(5+self.num_classes)
            current_anchor = out[...,ll:rl]
            current_anchor[...,0:2] = current_anchor[...,0:2].sigmoid()
            current_anchor[...,2:4] = F.softplus(current_anchor[...,2:4].clone())
            current_anchor[...,4:5] = current_anchor[...,4:5].sigmoid()
            anchors.append(current_anchor)
        final_pred = torch.cat(anchors, dim=-1)
        return final_pred




    def forward(self, x):
        out = self.detector(x).permute(0, 2, 3, 1).contiguous() # [BS, 7,7,170]
        return self._activate_output(out)

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.backbone = YOLOV1Backbone()
        self.head = YOLOV1Head(GRID_SIZE, NUM_CLASSES, 2)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
