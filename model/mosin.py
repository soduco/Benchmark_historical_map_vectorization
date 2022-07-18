import torch.nn as nn
from torchvision import models
import torch

class VGGNet(nn.Module):

    def __init__(self, name, layers):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.view(x.shape[0], -1))
        return results


class mosin(nn.Module):
    def __init__(self, unet, vggnet, args):
        super(mosin, self).__init__()
        self.unet = unet
        self.vggnet = vggnet
        self.K = args.K

    def forward(self, x, y):
        # x: images
        # y: init_labels
        results = []
        pred = []
        for i in range(self.K):
            input = torch.cat((x, y), dim=1)
            y = self.unet(input)
            # y = torch.sigmoid(y)
            pred.append(y)
        y_topo = self.vggnet(torch.cat((y, y, y), dim=1))
        results.append([pred, y_topo])
        return results


def choose_vgg(name):
    f = None
    if name == 'vgg11':
        f = models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained = True)

    for params in f.parameters():
        params.requires_grad = False

    return f