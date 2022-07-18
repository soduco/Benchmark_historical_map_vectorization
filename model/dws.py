import torch
import torch.nn as nn
from model.unet import *

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(
            m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(m.bias)

class watershed_net_combine(nn.Module):
    def __init__(self, mode='train', pretrain_weight_direction_path=None, output_classes=16):
        super(watershed_net_combine, self).__init__()
        self.output_classes = output_classes

        self.distance_net = UNET(n_channels=3, n_classes=2)
        self.watershed_net = UNET(n_channels=2, n_classes=16)
        if mode == 'train':
            print('Train mode')
            if pretrain_weight_direction_path:
                pretrain_weights_1 = torch.load(pretrain_weight_direction_path)
                self.distance_net.load_state_dict(pretrain_weights_1)
                print('-- Load pretrain weight in direction net: {} '.format(pretrain_weight_direction_path))
            else:
                print('No pretrain in direction network.')

            # If no pretrain, apply KAIMING initialization
            self.watershed_net.apply(weights_init)
            print('Kaiming initliazation done.')
        else:
            print('Inference mode')

    def forward(self, x):
        x = self.distance_net(x)
        output = self.watershed_net(x)
        return output
