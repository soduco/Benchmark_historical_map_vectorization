""" Full assembly of the parts to form the complete network """
from torch import nn
from model.unet_parts import *


class UNET(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, mode='original'):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if mode == 'mini':
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 512 // factor)
            self.up1 = Up(512, 256 // factor, bilinear)
            self.up2 = Up(256, 128 // factor, bilinear)
            self.up3 = Up(128, 64 // factor, bilinear)
            self.up4 = Up(64, 32, bilinear)
            self.outc = OutConv(32, n_classes)
        elif mode == 'original':
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _initialize_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.zeros_(m.bias)

        # # If no pretrain, apply KAIMING initialization
        self.apply(weights_init)
        print('Kaiming initliazation done.')
