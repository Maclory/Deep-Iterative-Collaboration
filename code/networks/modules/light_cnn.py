'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
    @modify: Steve Tod
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(3, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.classifier = nn.Sequential(mfm(8*8*128, 256, type=0), nn.LeakyReLU(0.2, True),
                nn.Linear(256, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class feature_extractor_9layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(
            nn.Sequential(mfm(3, 48, 5, 1, 2), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(48, 96, 3, 1, 1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(96, 192, 3, 1, 1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(192, 128, 3, 1, 1), 
                group(128, 128, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )

    def forward(self, x):
        feature_list = []
        for f in self.features:
            x = f(x)
            feature_list.append(x)
        return feature_list

def LightCNN_Feature_9Layers(**kwargs):
    model = feature_extractor_9layers(**kwargs)
    return model

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model
