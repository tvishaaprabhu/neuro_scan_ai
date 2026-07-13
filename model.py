"""Shared model definition — imported by both train.py and predict.py."""
import torch
import torch.nn as nn

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}
N_CLASSES = 4
IMG_SIZE = 128


def conv_block(in_ch, out_ch, dropout):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class BrainTumorCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.block1 = conv_block(1,   32,  0.20)   # 128 -> 64
        self.block2 = conv_block(32,  64,  0.25)   # 64  -> 32
        self.block3 = conv_block(64,  128, 0.30)   # 32  -> 16
        self.block4 = conv_block(128, 256, 0.30)   # 16  -> 8

        self.features = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

    def gradcam_target_layer(self):
        """Last conv layer before pooling — where Grad-CAM hooks in."""
        return self.block4[4]
