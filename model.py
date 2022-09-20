import torch.nn.functional as F
from params import args
from torch import nn
import torch as t

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform

class Encoder(nn.Module):
    def __init__(self, inpChannel=24):
        super(Encoder, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(inpChannel, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc = torch.nn.Linear(512, outClass)

        def forward(self, input_):
            input_ = self.layer0(input_)
            input_ = self.layer1(input_)
            input_ = self.layer2(input_)
            input_ = self.layer3(input_)
            input_ = self.layer4(input_)
            input_ = self.gap(input_)
            input_ = torch.flatten(input_)
            input_ = self.fc(input_)

            return input_

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_):
        shortcut = self.shortcut(input_)
        input_ = nn.ReLU()(self.bn1(self.conv1(input_)))
        input_ = nn.ReLU()(self.bn2(self.conv2(input_)))
        input_ = input_ + shortcut
        return nn.ReLU()(input_)
