import sys
sys.path.insert(1, '../')

import pandas as pd
import yaml
import json

from model.unet import UNET
from model.hed import hed
from model.bdcn import bdcn
from model.segmenter.factory import create_segmenter
from model.pvt import pvt_2
from model.dws import watershed_net_combine

import pdb


def load_config():
        return yaml.load(
        open('../config/config.yml', 'r'), Loader=yaml.FullLoader
    )

def model_param_count():
    model_unet = UNET(n_channels=3, n_classes=1)
    unet_count = sum(p.numel() for p in model_unet.parameters() if p.requires_grad)

    model_mini_unet = UNET(n_channels=3, n_classes=1, mode='mini')
    mini_unet_count = sum(p.numel() for p in model_mini_unet.parameters() if p.requires_grad)

    model_hed = hed(pretrain=True)
    hed_count = sum(p.numel() for p in model_hed.parameters() if p.requires_grad)

    model_bdcn = bdcn(pretrain=True)
    bdcn_count = sum(p.numel() for p in model_bdcn.parameters() if p.requires_grad)

    model_vit = create_segmenter(load_config()['net_kwargs'], mode='epm')
    vit_count = sum(p.numel() for p in model_vit.parameters() if p.requires_grad)

    model_pvt = pvt_2()
    pvt_count = sum(p.numel() for p in model_pvt.parameters() if p.requires_grad)

    model_watershed = watershed_net_combine('train')
    ws_count = sum(p.numel() for p in model_watershed.parameters() if p.requires_grad)
    model_summary = {
        'unet':  unet_count,
        'mini-unet': mini_unet_count,
        'hed' :  hed_count,
        'bdcn':  bdcn_count,
        'vit':   vit_count,
        'pvt_count': pvt_count,
        'ws_count': ws_count
    }
    with open('model_parm.json', 'w') as json_file:
        json.dump(model_summary, json_file)

if __name__ == '__main__':
    model_param_count()