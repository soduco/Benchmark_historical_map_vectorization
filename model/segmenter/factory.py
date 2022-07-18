
import torch
import os
from timm.models.vision_transformer import default_cfgs

from model.segmenter.vit import VisionTransformer
from model.segmenter.utils_model import checkpoint_filter_fn
from model.segmenter.decoder import DecoderLinear
from model.segmenter.decoder import MaskTransformer
from model.segmenter.segmenter import Segmenter
import timm

def create_segmenter(model_cfg, mode):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")

    encoder = create_vit(model_cfg, mode=mode)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

def create_new_vit():
    model = timm.create_model('vit_large_patch16_384', pretrained=True)
    return model

def create_vit(model_cfg, mode='epm'):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    if mode == 'ws':
        model_cfg["n_cls"] = 16
    else:
        model_cfg["n_cls"] = 1
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=True,
            num_classes=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    else:
        pass
    return model
