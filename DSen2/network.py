import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, feature_size, kernel_size, scale):
        super(ResBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size=kernel_size, padding='same', bias=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=kernel_size, padding='same', bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = out * self.scale
        out += x
        return out



class DSen2Model(nn.Module):
    def __init__(self, input_shape, num_layers=6, feature_size=128, kernel_size=3, scale=0.1):
        super(DSen2Model, self).__init__()
        self.num_layers = num_layers
        if len(input_shape) == 3:
            self.conv1 = nn.Conv2d(input_shape[0]+input_shape[1]+input_shape[2], feature_size, kernel_size=3, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(input_shape[0]+input_shape[1], feature_size, kernel_size=3, padding="same", bias=True)

        modules = []
        for i in range(num_layers):
            modules.append(ResBlock(feature_size, kernel_size, scale))
        self.resnet = nn.Sequential(*modules)

        self.conv2 = nn.Conv2d(feature_size, input_shape[-1], kernel_size=3, padding='same', bias=True)

    def forward(self, inputs_10, inputs_20, inputs_60=None):
        if inputs_60 != None:
            x = torch.cat([inputs_10, inputs_20, inputs_60], dim=1)
        else:
            x = torch.cat([inputs_10, inputs_20], dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.resnet(x)
        x = self.conv2(x)
        if inputs_60 != None:
            x = x + inputs_60
        else:
            x = x + inputs_20
        return x


