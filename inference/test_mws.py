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

# Import evaluation
from evaluation.eval_shape_detection import shape_detection
from evaluation.all_eval.run_eval import evaluation

import pdb


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, win_size, args):
    if args.dataset == 'atlas_municipal':
        input = args.original_image_path
        gt = args.gt_image_path
    elif args.dataset == 'verniquet':
        input = args.original_image_path_verniquet
        gt = args.gt_image_path_verniquet

    train_img  = Data(input, gt, win_size, args.unseen)
    testloader = torch.utils.data.DataLoader(train_img, batch_size=40, shuffle=False, num_workers=0, pin_memory=True)

    if args.cuda:
        model.to(args.device)

    model.eval()
    for i, (images, _) in enumerate(tqdm(testloader)):
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

    if args.dataset == 'atlas_municipal':
        res_dir = str(Path(args.model).parent.parent)
        output_dir = os.path.join(res_dir, '{}_test_evaluation'.format(str(args.model_type)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Reconstruct image
        tile_save_image_path = os.path.join(output_dir, '{}_test.png'.format(str(args.model_type)))

        in_img = cv2.imread(args.original_image_path)

        pad_px = win_size // 2
        new_img = reconstruct_from_patches(patches_images_ws, win_size, pad_px, in_img.shape, np.float32)

        new_img_ws = (new_img*255).astype(np.uint8)
        cv2.imwrite(tile_save_image_path, new_img_ws)
        # Boarder Calibration
        BOD = cv2.imread(args.EPM_border, 0)
        new_img_ws[BOD == 255] = 255
        new_img_ws = new_img_ws.astype(np.uint8)
        cv2.imwrite(tile_save_image_path, new_img_ws)
        print('Save reconstruction calibration image into {}'.format(tile_save_image_path))

        dynamic = args.dynamic
        area    = args.area
        output_path = os.path.join(output_dir, 'label_map.tif')
        meyer_watershed(tile_save_image_path, dynamic, area, output_path, './out.png')

        gt = cv2.imread(args.gt_image_path, 0)
        gt = (gt / 255).astype(np.uint8)

        pred_ws = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        correct, complete, quality, cldice, betti_p, betti_r, betti_f1 = evaluation(new_img, gt, pred_ws)

        iou_threshold = 0.5
        input_contenders_path = [output_path]
        precisions, recalls, f_score, iou_info, coco_matrix = shape_detection(args.gt_label_path, input_contenders_path, output_dir, iou_threshold, args.validation_mask)
        pq, sq, rq = coco_matrix['label_map']['PQ'], coco_matrix['label_map']['SQ'], coco_matrix['label_map']['RQ']
        test_df = pd.DataFrame({args.model_type :{
        'Correct'  : correct,
        'Complete' : complete,
        'Quality'  : quality,
        'ClDice'   : cldice,
        'B_p'      : betti_p,
        'B_r'      : betti_r,
        'B_f1'     : betti_f1,
        'pq_test'  : pq,
        'sq_test'  : sq,
        'rq_test'  : rq}
        })

        save_json = os.path.join(res_dir, 'test.json')
        test_df.to_json(save_json)
    elif args.dataset == 'verniquet':
        res_dir = os.path.join(str(Path(args.model).parent.parent), 'verniquet_eval_results')
        output_dir = os.path.join(res_dir, '{}_{}_test_evaluation'.format(str(args.series), str(args.model_type)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Reconstruct image
        tile_save_image_path = os.path.join(output_dir, '{}_test.png'.format(str(args.model_type)))
        in_img = cv2.imread(args.original_image_path_verniquet)

        pad_px = win_size // 2
        new_img = reconstruct_from_patches(patches_images_ws, win_size, pad_px, in_img.shape, np.float32)

        new_img_ws = (new_img*255).astype(np.uint8)
        cv2.imwrite(tile_save_image_path, new_img_ws)

        # Boarder Calibration
        BOD = cv2.imread(args.EPM_border_verniquet, 0)
        new_img_ws[BOD == 255] = 255
        new_img_ws = new_img_ws.astype(np.uint8)
        cv2.imwrite(tile_save_image_path, new_img_ws)
        print('Save reconstruction calibration image into {}'.format(tile_save_image_path))

        area, dynamic = int(args.model.split('/')[-1].split('_')[0]), int(args.model.split('/')[-1].split('_')[1])
        print('area: {}, dynamic: {}'.format(area, dynamic))

        output_path = os.path.join(output_dir, 'label_map.tif')
        meyer_watershed(tile_save_image_path, dynamic, area, output_path, './out.png')

        gt = cv2.imread(args.gt_image_path_verniquet, 0)
        gt = (gt / 255).astype(np.uint8)

        pred_ws = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        correct, complete, quality, cldice, betti_p, betti_r, betti_f1 = evaluation(new_img, gt, pred_ws)

        iou_threshold = 0.5
        input_contenders_path = [output_path]
        precisions, recalls, f_score, iou_info, coco_matrix = shape_detection(args.gt_label_path_verniquet, input_contenders_path, output_dir, iou_threshold, args.validation_mask_verniquet)
        pq, sq, rq = coco_matrix['label_map']['PQ'], coco_matrix['label_map']['SQ'], coco_matrix['label_map']['RQ']
        test_df = pd.DataFrame({args.model_type :{
        'Correct'  : correct,
        'Complete' : complete,
        'Quality'  : quality,
        'ClDice'   : cldice,
        'B_p'      : betti_p, 
        'B_r'      : betti_r,
        'B_f1'     : betti_f1,
        'pq_test'  : pq,
        'sq_test'  : sq,
        'rq_test'  : rq}
        })

        save_json = os.path.join(output_dir, 'test.json')
        test_df.to_json(save_json)

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
        sal_2_polygon(args.original_image_path, vector_path, res_dir)

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
    parser.add_argument('--dataset', type=str, default='verniquet',
                        help='The type dataset')
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

    parser.add_argument('--original_image_path', type=str,
                        default='../dataset/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-INPUT_color_border.jpg', help='Original image')
    parser.add_argument('--gt_image_path', type=str,
                        default='../dataset/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-EDGE_target.png', help='GT edges')
    parser.add_argument('--gt_label_path', type=str, 
                        default='../dataset/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-GT_LABELS_target.png', help='Gt labels')
    parser.add_argument('--validation_mask', type=str, 
                        default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-MASK_content.png', help='Validation mask to evaluate the results')
    parser.add_argument('--EPM_border', type=str, 
                        default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-EPM-BORDER-MASK_content.png', help='The mask of the EPM_border')

    parser.add_argument('--series', type=int, default=37,
                    help='Series value of verniquet datset')
    parser.add_argument('--original_image_path_verniquet', type=str,
                        default='../dataset/image_gt_verniquet/101100{}.png'.format(parser.parse_args().series), help='Original image')
    parser.add_argument('--gt_image_path_verniquet', type=str,
                        default='../dataset/image_gt_verniquet/Verniquet_planche_{}_GT_EDGE_target.tif'.format(parser.parse_args().series), help='GT edges')
    parser.add_argument('--gt_label_path_verniquet', type=str, 
                        default='../dataset/image_gt_verniquet/Verniquet_planche_{}_GT_LABELS_target.png'.format(parser.parse_args().series), help='Gt labels')
    parser.add_argument('--validation_mask_verniquet', type=str, 
                        default=r'../dataset/image_gt_verniquet/Verniquet_planche_{}_MASK_content.tif'.format(parser.parse_args().series), help='Validation mask to evaluate the results')
    parser.add_argument('--EPM_border_verniquet', type=str, 
                        default=r'../dataset/image_gt_verniquet/Verniquet_planche_{}_GT_EPM-BORDER-MASK_content.tif'.format(parser.parse_args().series), help='The mask of the EPM_border')

    parser.add_argument('--vgg', type=str, default='vgg19',
						help='pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', default=[4, 9, 18], type=int,
						help='the extracted features from vgg [4, 9, 27]')
    parser.add_argument('--K', type=int, default=3,
						help='number of iterative steps')
    parser.add_argument('--mu', type=float, default=10,
						help='loss coeff for vgg features')

    return parser.parse_args()


if __name__ == '__main__':
    main()
