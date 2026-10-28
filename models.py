from constants import NUM_CLASSES

import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.dconv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.dconv(x)


class UNET(nn.Module):
  def __init__(self, in_channels=3, out_channels=NUM_CLASSES, features=(64, 128, 256, 512)):
    super().__init__()
    self.downs = nn.ModuleList()
    self.ups = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    self.ground = DoubleConv(features[-1], features[-1] * 2)
    decoder_channels = features[-1] * 2
    for feature in reversed(features):
      self.ups.append(nn.ConvTranspose2d(decoder_channels, feature, kernel_size=2, stride=2))
      self.ups.append(DoubleConv(feature * 2, feature))
      decoder_channels = feature
    self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections = []
    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = self.pool(x)

    x = self.ground(x)
    for i, skip in enumerate(reversed(skip_connections)):
      x = self.ups[2 * i](x)
      if x.shape[-2:] != skip.shape[-2:]:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
      x = self.ups[2 * i + 1](torch.cat((skip, x), dim=1))
    return self.output(x)
