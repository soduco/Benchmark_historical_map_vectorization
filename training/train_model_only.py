import sys
sys.path.insert(1, '../')

import os
import numpy as np
import torch
from torch import optim
import argparse
import time
import datetime
from tqdm.auto import tqdm
import yaml
import cv2
from pathlib import Path

# Import dataloader
from data.smart_data_loader import Data

# Import model
from model.unet import unet
from model.hed import hed
from model.bdcn import bdcn
from model.segmenter.factory import create_segmenter
from model.pvt import pvt_2
from model.dws import watershed_net_combine

# Import loss function
from loss.bce_loss import cross_entropy_loss2d_sigmoid
from loss.multi_scale_bce_loss import ms_bce_loss
from loss.distance_map_loss import distance_softmax

# Import Utils
from utils import log
from utils.reconstruct_tiling_dict import reconstruct_from_patches


def train(args):
    # Initialize the model
    label_type = None
    if args.model_type == 'unet':
        model = unet(n_channels=args.channels, n_classes=args.classes)
        w_size = 500
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
        model = pvt_2()
        w_size = 256
    elif args.model_type == 'deep_watershed':
        model = watershed_net_combine('train', args.pretrain_weight_path_direction)
        label_type = 'learned_watershed'
    else:
        pass
    print('Training with model: {}'.format(args.model_type))

    aug_mode = False
    if args.data_aug:
        aug_mode = args.data_aug_mode
        data_aug_stat = 'aug_' + aug_mode
    else:
        data_aug_stat = 'no_aug'
    print('Data augmentation: {}; mode: {}'.format(str(data_aug_stat), aug_mode))

    train_img = Data(args.train_original_image_path, args.train_gt_path, w_size, args.data_aug, aug_mode=aug_mode,  mode=label_type)
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
    n_train = len(trainloader)

    val_img = Data(args.val_original_image_path, args.val_gt_path, w_size, data_aug=None)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    n_val = len(valloader)

    # Change it to adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

    start_time = time.time()
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
    for epoch in range(start_epoch, start_epoch+epochs):
        model.train()
        mean_loss = []
        with tqdm(total=int(n_train*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (img, labels) in enumerate(trainloader):
                # Set the gradient in the model into 0
                optimizer.zero_grad()

                # If batchsize not equal to batch index , calculate the current loss
                if args.cuda:
                    img, labels  = img.cuda(), labels.cuda()

                out = model(img)
                if args.model_type == 'unet' or args.model_type == 'vit' or args.model_type == 'pvt':
                    bce_loss = cross_entropy_loss2d_sigmoid(out, labels)
                elif args.model_type == 'hed' or args.model_type == 'bdcn':
                    bce_loss = ms_bce_loss(out, labels, args.batch_size, args.model_type, args.side_weight, args.fuse_weight)
                elif args.model_type == 'deep_watershed':
                    dist = labels['distance_map']
                    bce_loss = distance_softmax(out, dist)

                total_loss = bce_loss

                # Back calculating loss
                bce_loss.backward()

                # update parameter, gradient descent, back propagation
                optimizer.step()

                mean_loss.append(total_loss.item())

                # Update the pbar
                pbar.update(labels.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{model_name + '_loss': bce_loss.item()})
            tm = time.time() - start_time
        train_mean_loss = np.mean(mean_loss)
        model.eval()
        val_mean_loss = []
        val_mean_bce_loss = []
        with tqdm(total=int(n_val*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (val_images, val_labels) in enumerate(valloader):
                if args.cuda:
                    val_images, val_labels  = val_images.cuda(),  val_labels.cuda()

                with torch.no_grad():
                    val_out = model(val_images)

                if args.model_type == 'unet' or args.model_type == 'vit' or args.model_type == 'pvt':
                    val_bce_loss = cross_entropy_loss2d_sigmoid(val_out, val_labels)
                elif args.model_type == 'hed' or args.model_type == 'bdcn':
                    val_bce_loss = ms_bce_loss(val_out, val_labels, args.batch_size, args.model_type, args.side_weight, args.fuse_weight)
                    val_out = torch.sigmoid(val_out[-1])
                elif args.model_type == 'deep_watershed':
                    val_dist = labels['distance_map']
                    val_bce_loss = distance_softmax(val_out, val_dist)
                    val_out = torch.softmax(val_out, 1)
                    val_out = np.argmax(val_out, 1)
                    val_out = (val_out == 0).type(torch.uint8)

                batch, _, _, _ = val_out.shape
                for index, b in enumerate(range(batch)):
                    fuse_ws = (val_out[b, ...]).cpu().numpy()[0,...]     # (500, 500)

                    if i == 0 and index == 0:
                        patches_images_ws = fuse_ws[np.newaxis,...]
                    else:
                        patches_images_ws = np.concatenate((patches_images_ws, fuse_ws[np.newaxis,...]), axis=0) # (1, 500, 500)

                val_total_loss = val_bce_loss
                val_mean_bce_loss.append(val_bce_loss.item())
                val_mean_loss.append(val_total_loss.item())

                # Update the pbar
                pbar.update(val_images.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{model_name + '_loss': val_bce_loss.item()})
            val_mean_total_loss_value = np.mean(val_mean_loss)

        logger.info('lr: %e, train_loss: %f, validation loss: %f, time using: %f' %
                    (optimizer.param_groups[0]['lr'],
                        torch.from_numpy(np.array(train_mean_loss)).cuda(),
                        torch.from_numpy(np.array(val_mean_total_loss_value)).cuda(),
                        tm))

        in_img = cv2.imread(args.val_original_image_path)
        pad_px = w_size // 2
        new_img = reconstruct_from_patches(patches_images_ws, w_size, pad_px, in_img.shape, np.float32)
        tile_save_image_path_ws = os.path.join('.', recon_save_path, str(epoch) + '_{}_reconstruct.png'.format(int(np.array(val_mean_total_loss_value))))

        new_img = (new_img*255).astype(np.uint8)
        BOD = cv2.imread(args.val_EPM_border, 0)
        new_img[BOD == 255] = 255

        cv2.imwrite(tile_save_image_path_ws, new_img)
        torch.save(model.state_dict(), '{}/topo_best_val_{}.pth'.format(parm_save_path, str(epoch)))  # Save best weight

        # Learning rate schedular to change learning
        scheduler.step(val_mean_total_loss_value)
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

    parser = argparse.ArgumentParser(
        description='Train leakage-loss for different args')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
                        help='the file to store log, default is log.txt')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--pretrain_weight_path_direction', type=path_exists, default='../pretrain_weight/checkpoint_direction_net.pth',
                        help='Pretrain weight for the direction (for validation)')
    parser.add_argument('-d', '--dataset', type=str,
                        default='HistoricalMap2020', help='The dataset to train')

    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-4,
                        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')

    parser.add_argument('--data_aug', action='store_true',
                        help='Augmentation the data or not')
    parser.add_argument('--data_aug_mode', type=str, default='bri+aff',
                        help='Augmentation mode')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size of one iteration, default 1')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='whether resume from some, default is None')

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

    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')
    parser.add_argument('--res_dir', type=str, default='../training_info/',
                        help='the dir to store result')

    parser.add_argument('--train_original_image_path', type=str, default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-INPUT_color_border.jpg', 
                        help='Validation image')
    parser.add_argument('--train_gt_path', type=str, default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-EDGE_target.png',
                        help='The ground truth of the gt path')
    parser.add_argument('--val_original_image_path', type=str,
                        default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_color_border.jpg', help='Validation image')
    parser.add_argument('--val_EPM_border', type=str,
                        default=r'../dataset/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-EPM-BORDER-MASK_content.png')
    parser.add_argument('--val_gt_path', type=str, default=r'../dataset/new_image_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png',
                        help='The ground truth of the gt path')

    return parser.parse_args()


if __name__ == '__main__':
    main()

