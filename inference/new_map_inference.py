import sys
sys.path.insert(1, '../')

import os
import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

# Import shapely -> geo tool
from shapely.ops import polygonize_full
from shapely.geometry import mapping
from skimage.measure import approximate_polygon
import fiona

# Import dataloader
from data.smart_data_loader import Data

# Import model
from model.unet import UNET
from model.hed import hed
from model.bdcn import bdcn
from model.segmenter.factory import create_segmenter
from model.pvt import pvt_2
from model.dws import watershed_net_combine
from model.mosin import mosin, VGGNet

# Import Utils
from utils.reconstruct_tiling_dict import reconstruct_from_patches


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, win_size, args):
    input = args.input_map_path
    gt = None

    # args.unseen == True -> gt is None
    test_img  = Data(input, gt, win_size, unseen=args.unseen)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=40, shuffle=False, num_workers=0, pin_memory=True)

    if args.cuda:
        model.to(args.device)

    model.eval()
    for i, (images) in enumerate(tqdm(testloader)):
        if args.cuda:
            images = images.to(args.device)

        with torch.no_grad():
            if args.model_type == 'mosin':
                init_labels = torch.zeros_like(images.shape[:2])
                init_labels = init_labels.to(args.device)
                out = model(images, init_labels)
                fuse = out[0][0][-1].cpu().numpy()
            elif args.model_type == 'hed' or args.model_type == 'bdcn' or args.model_type == 'hed_pretrain' or args.model_type == 'bdcn_pretrain':
                out = model(images)
                fuse = torch.sigmoid(out[-1]).cpu().numpy()
            elif args.model_type == 'dws':
                out = model(images)
                out = torch.softmax(out, 1).squeeze()
                out = torch.argmax(out, 0)
                fuse = (out == 0).type(torch.uint8).cpu().numpy()
            else:
                out = model(images)
                fuse = torch.sigmoid(out).cpu().numpy()

        if i == 0:
            patches_images_ws = fuse[:, 0,...]
        else:
            patches_images_ws = np.concatenate((patches_images_ws, fuse[:, 0,...]), axis=0)
    return patches_images_ws


def meyer_watershed(image_path, dynamic, area, output_path, out_visu_path):
    print('../watershed/histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))
    os.system('../watershed/histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Choose the GPUs
    args.device = torch.device("cuda:0".format(str(args.gpu)) if torch.cuda.is_available() else "cpu")

    if args.model_type == 'unet' or args.model_type == 'bal' or args.model_type == 'topo' or args.model_type == 'pathloss' or args.model_type == 'unet_bri' or args.model_type == 'unet_aff' or args.model_type == 'unet_hom' or args.model_type == 'unet_tps' or args.model_type == 'unet_bri_aff' or args.model_type == 'unet_bri_hom' or args.model_type == 'unet_bri_tps':
        model = UNET(n_channels=args.channels, n_classes=args.classes)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'mini-unet':
        model = UNET(n_channels=args.channels, n_classes=args.classes, mode='mini')
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'vit':
        def load_config():
            return yaml.load(open('../config/config.yml', 'r'), Loader=yaml.FullLoader)
        model_cfg = load_config()['net_kwargs']
        model_cfg['dropout'] = 0.0
        model = create_segmenter(model_cfg, mode='epm')
        pretrain_weight = torch.load('%s' % (args.model))
        model.load_state_dict(pretrain_weight)
        print('Load model {}'.format(args.model))
        win_size = 256
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'mosin':
        model = UNET(n_channels=args.channels, n_classes=args.classes)
        vggnet = VGGNet(args.vgg, args.layers)
        model = mosin(model, vggnet, args)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'pvt':
        model = pvt_2()
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 256
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'hed' or args.model_type == 'hed_pretrain':
        model = hed(pretrain=None)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'bdcn' or args.model_type == 'bdcn_pretrain':
        model = bdcn(pretrain=None)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)
    elif args.model_type == 'dws':
        model = watershed_net_combine('Inference')
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
        patches_images_ws = test(model, win_size, args)

    output_dir = os.path.join(str(Path(args.model).parent), 'epm_eval_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reconstruct image
    tile_save_image_path = os.path.join(output_dir, '{}_test.png'.format(str(args.model_type)))
    in_img = cv2.imread(args.input_map_path)

    pad_px = win_size // 2
    new_img = reconstruct_from_patches(patches_images_ws, win_size, pad_px, in_img.shape, np.float32)

    new_img_ws = (new_img*255).astype(np.uint8)
    cv2.imwrite(tile_save_image_path, new_img_ws)
    print('Save reconstruction calibration image into {}'.format(tile_save_image_path))

    output_path = os.path.join(output_dir, 'label_map.tif')
    meyer_watershed(tile_save_image_path, args.dynamic, args.area, output_path, './out.png')

    if args.vectorization:
        save_output = os.path.join(output_dir, 'ws_output')
        if not os.path.exists(save_output): os.makedirs(save_output)

        save_vector = os.path.join(output_dir, 'vector_output')
        if not os.path.exists(save_vector): os.makedirs(save_vector)

        output_path = os.path.join(save_output, '{}.tiff'.format('label'))
        vector_path = os.path.join(save_vector, '{}.npy'.format('vector_lines'))
        out_visu_path = os.path.join(output_dir, 'out.png')
        print('/lrde/home2/ychen/hierarchy_watershed/temporar_python_bindings/vectorization_ws_meyer/build/histmapseg {} {} {} {} {} {}'.format(tile_save_image_path, int(args.dynamic), int(args.area), output_path, out_visu_path, vector_path))
        os.system('/lrde/home2/ychen/hierarchy_watershed/temporar_python_bindings/vectorization_ws_meyer/build/histmapseg {} {} {} {} {} {}'.format(tile_save_image_path, int(args.dynamic), int(args.area), output_path, out_visu_path, vector_path))
        sal_2_polygon(args.original_image_path, vector_path, output_dir)

    print('Done')

def di(lines, i):
    return lines[lines[..., 2] == i, :2]

def sal_2_polygon(img, vector_path, res_dir, dp_tol=2):
    lines = np.load(vector_path)
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    save_vector_path = os.path.join(res_dir, 'shape_file')
    if not os.path.exists(save_vector_path): 
        os.makedirs(save_vector_path)

    lineShp = fiona.open(os.path.join(save_vector_path, 'Polygon.shp'), mode='w', driver='ESRI Shapefile', schema = schema, crs = "EPSG:4326")
    x, y = img.shape
    all_lines = []

    # Add border lines
    left_top     = (0,  0)
    left_bottom  = (0,  -x)
    right_top    = (y, 0)
    right_bottom = (y, -x)

    for i in np.unique(lines[..., 2]):
        d_plot = di(lines, i)
        d_plot = approximate_polygon(d_plot, tolerance=dp_tol)
        all_lines += [((int(d_plot[d][0]/2), -int(d_plot[d][1]/2)), (int(d_plot[d+1][0]/2), -int(d_plot[d+1][1]/2))) for d in range(0, len(d_plot)-1)]
    
    # Add border lines
    all_lines.append((left_top, left_bottom))
    all_lines.append((left_bottom, right_bottom))
    all_lines.append((right_bottom, right_top))
    all_lines.append((right_top, left_top))
    result, dangles, cuts, invalids = polygonize_full(all_lines)

    print('Number of valid geometry: {}'.format(len(result.geoms)))
    print('Number of dangles geometry: {}'.format(len(dangles.geoms)))
    print('Number of cuts geometry: {}'.format(len(cuts.geoms)))
    print('Number of invalids geometry: {}'.format(len(invalids.geoms)))
    
    for index in range(0, len(result.geoms)):
        lineShp.write({
            'geometry': mapping(result.geoms[index]),
            'properties': {'id': index},
        })

    lineShp.close()

def parse_args():
    parser = argparse.ArgumentParser('Test UNET')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--unseen', action='store_true',
                        help='Unseen dataset')
    parser.add_argument('--vectorization', action='store_true',
                        help='Vectorization the maps')

    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str,
                        default=None, help='the model to test')

    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')

    parser.add_argument('-a', '--area', type=int, default=100,
                        help='Area of the meyer watershed.')
    parser.add_argument('-d', '--dynamic', type=int, default=7,
                        help='Dynamics of the meyer watershed.')

    parser.add_argument('--vgg', type=str, default='vgg19',
						help='pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', default=[4, 9, 18], type=int,
						help='the extracted features from vgg [4, 9, 27]')
    parser.add_argument('--K', type=int, default=3,
						help='number of iterative steps')
    parser.add_argument('--mu', type=float, default=10,
						help='loss coeff for vgg features')

    parser.add_argument('--input_map_path', type=str,
                        default='', help='Input map image.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
