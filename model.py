# model.py
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=11, pretrained=True):

    net = models.resnet50(pretrained=pretrained)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p = 0.4),
        nn.Linear(in_features, num_classes)
    )
    return net