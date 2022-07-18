import sys
sys.path.insert(1, '../')

import os
import numpy as np
import torch
from torch import optim
from torch import nn
import argparse
import time
import datetime
from tqdm.auto import tqdm
import yaml
import cv2

# Import dataloader
from data.smart_data_loader import Data

# Import config
import config.cfg as cfg

# Import model
from model.unet import unet
from model.hed import hed
from model.bdcn import bdcn
from model.segmenter.factory import create_segmenter
from model.pvt import pvt
from model.mosin import mosin, VGGNet

# Import loss function
from loss.bce_loss import cross_entropy_loss2d_sigmoid
from loss.multi_scale_bce_loss import ms_bce_loss
from loss.mosin_loss import iterative_loss
from loss.topo_loss import getTopoLoss
from loss.MBD_BAL.BALoss import boundary_awareness_loss
from loss.path_loss.p_loss import Path_loss

# Import Utils
from utils import log
from utils.reconstruct_tiling_dict import reconstruct_from_patches


def train(args):
    # Initialize the model 
    if args.model_type == 'unet':
        model = unet(n_channels=args.channels, n_classes=args.classes)
        w_size = 500
        if args.topo_loss_type == 'topoloss' or args.topo_loss_type == 'baloss' or args.topo_loss_type == 'pathloss':
            model.load_state_dict(torch.load(args.pretrain))
            print('Load pretrain: {}'.format(args.pretrain))
    elif args.model_type == 'hed':
        model = hed()
        w_size = 500
    elif args.model_type == 'bdcn':
        model = bdcn()
        w_size = 500
    elif args.model_type == 'vit':
        pretrain = True
        def load_config():
            return yaml.load(
            open('../config/config.yml', 'r'), Loader=yaml.FullLoader
        )
        model_cfg = load_config()['net_kwargs']
        model = create_segmenter(model_cfg, mode='epm')
        if pretrain:
            pretrain_weights_vit = torch.load('../pretrain_weight/checkpoint_vit.pth')['model']
            del pretrain_weights_vit['encoder.head.weight']
            del pretrain_weights_vit['encoder.head.bias']
            del pretrain_weights_vit['decoder.head.weight']
            del pretrain_weights_vit['decoder.head.bias']
            model.load_state_dict(pretrain_weights_vit, strict=False)
        w_size = 256
    elif args.model_type == 'pvt':
        model = pvt()
        w_size = 256
    elif args.model_type == 'mosin':
        model = unet(n_channels=4, n_classes=args.classes)
        model = model.cuda()
        vggnet = VGGNet(args.vgg, args.layers)
        model = mosin(model, vggnet, args)
        pretrain_path = '/lrde/work/ychen/PRL/benchmark_DL/unet_original/HistoricalMap2020/mosin_unet/2022-04-20_23:28:32_lr_0.0001_train_unet_orign_bs_1/params/topo_best_val_11.pth'
        pretrain_weight = torch.load(pretrain_path)
        for index, key in enumerate(list(pretrain_weight.keys())):
            if key.split('.')[0] == 'UNet':
                new_key = key.replace('UNet', 'unet')
                pretrain_weight[new_key] = pretrain_weight.pop(key)
        model.load_state_dict(pretrain_weight)
        print('Load weight unet pretrain {}'.format(pretrain_path))
        w_size = 500
    else:
        pass

    if not(args.topo_loss_type):
        args.topo_loss_type = 'BCE_loss'

    print('Training with model: {}'.format(args.model_type))
    print('Training with loss:  {}'.format(args.topo_loss_type))

    aug_mode = args.data_aug_mode
    if args.data_aug:
        data_aug_stat = 'aug_' + aug_mode
    else:
        data_aug_stat = 'no_aug'

    train_img_path = '/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-INPUT_color_border.jpg'
    train_gt_path  = '/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-EDGE_target.png'
    train_img = Data(train_img_path, train_gt_path, w_size, args.data_aug, aug_mode=aug_mode, dilation=True, mode='loss')
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
    n_train = len(trainloader)

    # Validation evaluation
    val_img_path = '/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_color_border.jpg'
    val_gt_path  = '/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-EDGE_target.png'
    val_img = Data(val_img_path, val_gt_path, w_size, data_aug=None, dilation=True, mode='loss')
    valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    n_val = len(valloader)

    # Change it to adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

    if args.cuda:
        model.cuda()

    if args.resume:
        model_pretrain = torch.load(args.resume)
        model.load_state_dict(model_pretrain)
        print('Resume pretrain {}'.format(args.resume))

        res_dir = Path(args.resume).parent.parent
        logger = log.get_logger(os.path.join(res_dir, '{}.txt'.format(args.model_type)), mode='a')
        start_epoch = int(args.resume.split('_')[-1].split('.')[0]) + 1
        recon_save_path = os.path.join(res_dir, 'reconstruction_png')
        parm_save_path = os.path.join(res_dir, 'params')
    else:
        model_name = args.model_type
        loss_type = 'train_{}'.format(model_name) + '_bs_'+ str(args.batch_size)

        # Create res directory
        res_dir = os.path.join(args.res_dir + args.dataset, model_name, str(datetime.datetime.now()).replace(' ', '_').split('.')[0] + '_lr_' + str(args.base_lr)) + '_' + loss_type + '_' + data_aug_stat
        print('Model save in {}'.format(res_dir))

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        recon_save_path = os.path.join(res_dir, 'reconstruction_png')
        if not os.path.exists(recon_save_path):
            os.makedirs(recon_save_path)

        # Create params folder
        parm_save_path = os.path.join(res_dir, 'params')
        if not os.path.exists(parm_save_path):
            os.makedirs(parm_save_path)

        # Create Logger 
        logger = log.get_logger(os.path.join(res_dir, '{}.txt'.format(args.model_type)))

        start_epoch = 0

    epochs = args.epochs
    bce_loss = 0
    topo_loss = 0
    val_bce_loss = 0
    val_topo_loss = 0
    for epoch in range(start_epoch, start_epoch+epochs):
        model.train()
        mean_loss = []
        mean_bce_loss= []
        mean_topo_loss = []
        with tqdm(total=int(n_train*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (img, labels, seeds) in enumerate(trainloader):
                # Set the gradient in the model into 0
                optimizer.zero_grad()

                # If batchsize not equal to batch index , calculate the current loss
                if args.cuda:
                    img, labels, seeds  = img.cuda(), labels.cuda(), seeds.cuda()

                if args.model_type == 'mosin':
                    init_labels = torch.zeros_like(labels)
                    init_labels = init_labels.cuda()
                    out = model(img, init_labels)
                else:
                    out = model(img)

                if args.model_type == 'unet' or args.model_type == 'vit' or args.model_type == 'pvt':
                    bce_loss = cross_entropy_loss2d_sigmoid(out, labels)
                elif args.model_type == 'hed' or args.model_type == 'bdcn':
                    bce_loss = ms_bce_loss(out, labels, args.batch_size, args.model_type, args.side_weight, args.fuse_weight)
                    out = out[-1]

                topo_loss = 0
                if args.topo_loss_type == 'mosin':
                    vgg_labels = model.vggnet(torch.cat((labels, labels, labels), dim=1))
                    bce_loss, _, topo_loss = iterative_loss(out, vgg_labels, labels, args)
                elif args.topo_loss_type == 'topoloss':
                    # Accumulate topoloss
                    for b in range(args.batch_size):
                        out_tmp = out[b].unsqueeze(0)
                        labels_tmp = labels[b].unsqueeze(0)
                        topo_loss += args.alpha * getTopoLoss(out_tmp, labels_tmp, topo_size=50)
                elif args.topo_loss_type == 'baloss':
                    for b in range(args.batch_size):
                        out_tmp = out[b].unsqueeze(0)
                        labels_tmp = labels[b].unsqueeze(0)
                        seeds_tmp = seeds[b]
                        topo_loss  += args.alpha * boundary_awareness_loss(out_tmp, seeds_tmp, labels_tmp)
                elif args.topo_loss_type == 'pathloss':
                    for b in range(args.batch_size):
                        out_tmp = out[b].unsqueeze(0)
                        labels_tmp = labels[b].unsqueeze(0)
                        seeds_tmp = seeds[b]
                        topo_loss += args.alpha * Path_loss(out_tmp, seeds_tmp, labels_tmp)
                else:
                    pass

                total_loss = bce_loss + topo_loss

                # Back calculating loss
                total_loss.backward()

                # update parameter, gradient descent, back propagation
                optimizer.step()

                mean_loss.append(total_loss.item())
                mean_bce_loss.append(bce_loss.item())
                mean_topo_loss.append(topo_loss.item())

                # Update the pbar
                pbar.update(labels.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'total_loss': total_loss.item(), 'topo_loss': topo_loss.item(), 'bce_loss': bce_loss.item()})
            train_mean_loss = np.mean(mean_loss)
            train_bce_loss = np.mean(mean_bce_loss)
            train_topo_loss = np.mean(mean_topo_loss)

        model.eval()
        val_mean_loss = []
        val_mean_bce_loss = []
        val_mean_topo_loss = []
        with tqdm(total=int(n_val*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (val_img, val_labels, val_seeds) in enumerate(valloader):
                if args.cuda:
                    val_img, val_labels, val_seeds  = val_img.cuda(), val_labels.cuda(), val_seeds.cuda()

                with torch.no_grad():
                    if args.model_type == 'mosin':
                        val_init_labels = torch.zeros_like(val_labels)
                        val_init_labels = val_init_labels.cuda()
                        val_out = model(val_img, val_init_labels)
                    else:
                        val_out = model(val_img)

                if args.model_type == 'unet' or args.model_type == 'vit' or args.model_type == 'pvt':
                    val_bce_loss = cross_entropy_loss2d_sigmoid(val_out, val_labels)
                elif args.model_type == 'hed' or args.model_type == 'bdcn':
                    val_bce_loss = ms_bce_loss(val_out, val_labels, args.batch_size, args.model_type, args.side_weight, args.fuse_weight)
                    val_out = val_out[-1]

                val_topo_loss = 0
                if args.topo_loss_type == 'mosin':
                    val_vgg_labels = model.vggnet(torch.cat((val_labels, val_labels, val_labels), dim=1))
                    val_bce_loss, _, val_topo_loss = iterative_loss(val_out, val_vgg_labels, val_labels, args)
                    val_out = val_out[0][0][-1]
                elif args.topo_loss_type == 'topoloss':
                    for b in range(args.batch_size):
                        val_out_tmp = val_out[b].unsqueeze(0)
                        val_labels_tmp = val_labels[b].unsqueeze(0)
                        val_topo_loss += args.alpha * getTopoLoss(val_out_tmp, val_labels_tmp, topo_size=50)
                elif args.topo_loss_type == 'baloss':
                    for b in range(args.batch_size):
                        val_out_tmp = val_out[b].unsqueeze(0)
                        val_labels_tmp = val_labels[b].unsqueeze(0)
                        val_seeds_tmp = val_seeds[b].unsqueeze(0)
                        val_topo_loss += args.alpha * boundary_awareness_loss(val_out_tmp, val_seeds_tmp, val_labels_tmp)
                elif args.topo_loss_type == 'pathloss':
                    for b in range(args.batch_size):
                        val_out_tmp = val_out[b].unsqueeze(0)
                        val_labels_tmp = val_labels[b].unsqueeze(0)
                        val_seeds_tmp = val_seeds[b].unsqueeze(0)
                        val_topo_loss += args.alpha * Path_loss(val_out_tmp, val_seeds_tmp, val_labels_tmp)
                else:
                    pass

                val_out = torch.sigmoid(val_out)
                batch, _, _, _ = val_out.shape
                for index, b in enumerate(range(batch)):
                    fuse_ws = (val_out[b, ...]).cpu().numpy()[0,...]

                    if i == 0 and index == 0:
                        patches_images_ws = fuse_ws[np.newaxis,...]
                    else:
                        patches_images_ws = np.concatenate((patches_images_ws, fuse_ws[np.newaxis,...]), axis=0) # (1, 500, 500)

                val_total_loss = val_bce_loss + val_topo_loss

                val_mean_loss.append(val_total_loss.item())
                val_mean_bce_loss.append(val_bce_loss.item())
                val_mean_topo_loss.append(val_topo_loss.item())

                # Update the pbar
                pbar.update(val_img.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'val_total_loss': val_total_loss.item(), 'val_topo_loss': val_topo_loss.item(), 'val_bce_loss': val_bce_loss.item()})
            val_mean_loss = np.mean(val_mean_loss)
            val_bce_loss = np.mean(val_mean_bce_loss)
            val_topo_loss = np.mean(val_mean_topo_loss)
        logger.info('lr: %e, train_total_loss: %f, train_bce_loss: %f, train_topo_loss: %f, val_total_loss: %f, val_bce_loss: %f, val_topo_loss: %f' %
                    (optimizer.param_groups[0]['lr'],
                        torch.from_numpy(np.array(train_mean_loss)).cuda(),
                        torch.from_numpy(np.array(train_bce_loss)).cuda(),
                        torch.from_numpy(np.array(train_topo_loss)).cuda(),
                        torch.from_numpy(np.array(val_mean_loss)).cuda(),
                        torch.from_numpy(np.array(val_bce_loss)).cuda(),
                        torch.from_numpy(np.array(val_topo_loss)).cuda(),
                        )
                    )

        in_img = cv2.imread(args.val_original_image_path)
        pad_px = w_size // 2
        new_img = reconstruct_from_patches(patches_images_ws, w_size, pad_px, in_img.shape, np.float32)
        tile_save_image_path_ws = os.path.join('.', recon_save_path, str(epoch) + '_{}_reconstruct.png'.format(int(np.array(val_mean_loss))))

        new_img = (new_img*255).astype(np.uint8)
        BOD = cv2.imread(args.val_EPM_border, 0)
        new_img[BOD == 255] = 255

        cv2.imwrite(tile_save_image_path_ws, new_img)
        torch.save(model.state_dict(), '{}/topo_best_val_{}.pth'.format(parm_save_path, str(epoch)))  # Save best weight

        # Learning rate schedular to change learning
        scheduler.step(val_mean_loss)
        print('Current learning rate             {}'.format(optimizer.param_groups[0]['lr']))


def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)

    train(args)

def parse_args():
    def path_exists(p):
        if(os.path.exists(p)):
            return p
        else:
            return None

    AUC_THRESHOLD_DEFAULT = 0.5

    parser = argparse.ArgumentParser(
        description='Train leakage-loss for different args')
    parser.add_argument('-p', '--pretrain', type=path_exists, default='../pretrain_weight/unet_best_pretrain.pth',
        help='init net from pretrained model default is None')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
                        help='the file to store log, default is log.txt')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--topo_loss_type', type=str, default=None,
                        help='The type of the model')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='the alpha')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_BAL_train.keys(),
                        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--param_dir', type=str, default='params',
                        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-4,
                        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('--data_aug', action='store_true',
                        help='Augmentation the data or not')
    parser.add_argument('--data_aug_mode', type=str, default='bri+aff',
                        help='Augmentation mode')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='whether resume from some, default is None')
    parser.add_argument('--model', type=str, default=None,
                        help='Pre-load model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch to train network, default is 100')
    parser.add_argument('--max-iter', type=int, default=40000,
                        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=10,
                        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
                        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1,
                        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=50,
                        help='the number of iters to decrease the learning rate, default is 50')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
                        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-k', type=int, default=1,
                        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
                        help='the size of image to crop, default not crop')
    parser.add_argument('--complete-pretrain', type=str, default=None,
                        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
                        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
                        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the decay of learning rate, default 0.1')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')
    parser.add_argument('--res_dir', type=str, default='../training_info/',
                        help='the dir to store result')
    parser.add_argument('--auc-threshold', type=float,
                        help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)
    parser.add_argument('--EPM_threshold', type=int, default=0.5,
                        help='Threshold to create binary image of EPM')
    parser.add_argument('--validation_mask', type=str, default=r'/lrde/image/CV_2021_yizi/historical_map_2020/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-MASK_content.png',
                        help='Validation mask to evaluate the results')
    parser.add_argument('--val_original_image_path', type=str,
                        default=r'/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_color_border.jpg', help='Validation image')
    parser.add_argument('--val_EPM_border', type=str,
                        default=r'/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/epm_mask/BHdV_PL_ATL20Ardt_1926_0004-VAL-EPM-BORDER-MASK_content.png')
    parser.add_argument('--val_gt_path', type=str, default=r'/lrde/home2/ychen/deep_watershed/new_image_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png',
                        help='The ground truth of the gt path')
    parser.add_argument('--vgg', type=str, default='vgg19',
						help='pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', default=[4, 9, 18], type=int,
						help='the extracted features from vgg [4, 9, 18, 27]')
    parser.add_argument('--K', type=int, default=3,
						help='number of iterative steps')
    parser.add_argument('--mu', type=float, default=10,
						help='loss coeff for vgg features')

    return parser.parse_args()


if __name__ == '__main__':
    main()

