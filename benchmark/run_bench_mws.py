# Run Benchmark
import sys
sys.path.insert(1, '../')

import numpy as np
import os
import cv2
import time

import argparse
from functools import partial
import pathos.pools as pp
import itertools
from pathlib import Path

from evaluation.eval_shape_detection import shape_detection


def tif2png(file_dir, EPM_border_file):
    save_dir = Path(file_dir).parent
    save_dir = os.path.join(str(save_dir), 'reconstruction_png')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        file_names = os.listdir(file_dir)
        for file in file_names:
            file_path = os.path.join(file_dir, file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = (image * 255).astype(np.uint8)
            BOD = cv2.imread(EPM_border_file, 0)
            image[BOD == 255] = 255

            new_file = os.path.join(save_dir, file.split('.')[0]+'.png')
            cv2.imwrite(new_file, image)
            print('Save reconstruction calibration image into {}'.format(new_file))
    else:
        print('Folder already created.')

    return save_dir

def meyer_watershed(image_path, dynamic, area, output_path, out_visu_path):
    print('../watershed/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))
    os.system('../watershed/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))

def save_label_maps(labels, output_path, model, mode, file_name, attribute, name):
    output_dir = os.path.join(output_path, model, mode, attribute, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    if not cv2.imwrite(output_path, labels.astype(np.uint16)):
        raise Exception("Could not write image")

def pre_save_label_maps(image_path, output_path, list_dynamic, list_area, model, mode):
    def save_label_maps_batch(image_path, output_path, parameters):
        dynamic, area = parameters
        file_name = image_path.split('/')[-1].split('.')[0]+'_'+str(dynamic)+'_'+str(area)+'.png'
        name = image_path.split('/')[-1].split('.')[0]
        output_dir = os.path.join(output_path, model, mode, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        meyer_watershed(image_path, dynamic, area, output_path, './out.png')

    test_list = list(itertools.product(list_dynamic, list_area))
    pool = pp.ProcessPool(nodes=10) # Pool
    get_partial = partial(save_label_maps_batch, image_path, output_path)
    results = pool.amap(get_partial, test_list)  # do an asynchronous map, then get the results
    result = results.get()

def grid_search_eval(gt_path, input_contenders_dir, output_dir, iou_threshold, input_mask, model, mode, name):
    # for input_contenders_path_name in file_names:
    def coco_eval_batch(input_contenders_path_name):
        output_val_dir = os.path.join(output_dir, model, mode, name, input_contenders_path_name.split('.')[0])

        if not os.path.exists(output_val_dir):
            os.makedirs(output_val_dir)
        input_contenders_path = os.path.join(input_contenders_dir, input_contenders_path_name)
        input_contenders_path = [input_contenders_path]
        _, _, _, _, coco_matrix = shape_detection(gt_path, input_contenders_path, output_val_dir, iou_threshold, input_mask, save_fig=False)
        dynamics, area = input_contenders_path_name.split('.')[0].split('_')[-2], input_contenders_path_name.split('.')[0].split('_')[-1]
        return [coco_matrix[input_contenders_path_name.split('.')[0]]['PQ'], coco_matrix[input_contenders_path_name.split('.')[0]]['RQ'], coco_matrix[input_contenders_path_name.split('.')[0]]['SQ'], dynamics, area]

    input_contenders_dir = os.path.join(input_contenders_dir, model, mode, name)
    file_names = os.listdir(input_contenders_dir)
    pool = pp.ProcessPool(nodes=10) # Pool
    get_partial = partial(coco_eval_batch)
    results = pool.amap(get_partial, file_names) # do an asynchronous map, then get the results
    coco_list = results.get()
    coco_list = np.array(coco_list)
    return coco_list, np.argmax(coco_list[:, 0].astype(float)), np.max(coco_list[:, 0].astype(float))

if __name__ == '__main__':
    def parse_args():
        AUC_THRESHOLD_DEFAULT = 0.5
        parser = argparse.ArgumentParser(description='Benchmark evaluaion')
        parser.add_argument('--dataset', type=str, default='hist',
            help='Types of dataset')
        parser.add_argument('--image_dir', type=str, default=r'/lrde/home2/ychen/release_code/release_code/training_info_ambiguous/HistoricalMap2020/unet/2022-11-27_17:36:05_lr_0.0001_train_unet_bs_1_no_aug_baseline/reconstruction_png',
            help='The input path of the image')

        parser.add_argument('--last_epoch', type=int, default=50,
            help='Last epochs ')
        ABS_PATH = Path(parser.parse_args().image_dir).parent
        parser.add_argument('--output_path', type=str, default=r'{}/label_maps_new_mws/'.format(ABS_PATH),
            help='The output path of the image')
        parser.add_argument('--input_contenders_dir', type=str, default=r'{}/label_maps_new_mws/'.format(ABS_PATH),
            help='Save contender path')
        parser.add_argument('--output_dir', type=str, default=r'{}/eval_folder_new_mws/'.format(ABS_PATH),
            help='Save evaluation path')
        parser.add_argument('--gt_path', type=str, default=r'/lrde/work/ychen/PRL/benchmark_mws/gt_label_path/gt_label_map.png'.format(ABS_PATH),
            help='The ground truth of the label path')
        parser.add_argument('--auc-threshold', type=float,
            help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)


        parser.add_argument('--validation_mask', type=str, default=r'/lrde/image/CV_2021_yizi/historical_map_2020/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-MASK_content.png',
            help='Validation mask to evaluate the results')
        parser.add_argument('--gt_label_path', type=str, default='/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png',
            help='the gt label path')
        parser.add_argument('--EPM_border', type=str, default=r'/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/epm_mask/BHdV_PL_ATL20Ardt_1926_0004-VAL-EPM-BORDER-MASK_content.png',      help='The mask of the EPM_border')
        return parser.parse_args()

    print('##########################  GRID SEARCH VALIDATION  ##################################')
    start = time.time()
    args = parse_args()
    mode = 'val'
    attribute = 'area'
    model = (args.image_dir).split('/')[-2]

    list_dynamic   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    list_area      = [50, 100, 200, 300, 400, 500]
    analysis_range = 1
    analysis_last_epoch = args.last_epoch

    save_dir = tif2png(args.image_dir, args.EPM_border)
    image_lst = os.listdir(save_dir)[:analysis_last_epoch]
    image_lst = image_lst[::analysis_range]
    image_lst.append(os.listdir(save_dir)[-1]) # Append the last file path

    best_parm_lst = []
    best_coco_lst = []
    full_cooo_lst = []

    for index, l in enumerate(image_lst):
        image_path = os.path.join(save_dir, l)
        name = image_path.split('/')[-1].split('.')[0]
        pre_save_label_maps(image_path, args.output_path, list_dynamic, list_area, model, mode)
        coco_list, best_param, max_coco = grid_search_eval(args.gt_path, args.input_contenders_dir, args.output_dir, args.auc_threshold, args.validation_mask, model, mode, name)
        best_parm_lst.append(best_param)
        best_coco_lst.append(max_coco)
        full_cooo_lst.append(coco_list)

    best_param = np.argmax(best_coco_lst)
    best_coco_array = np.array(list(map(float, full_cooo_lst[best_param][:,0])))
    best_coco_index =  np.argmax(best_coco_array)
    best_coco, best_sq, best_rq, best_dynamics, best_area = full_cooo_lst[best_param][best_coco_index]
    best_coco, best_sq, best_rq = round(float(best_coco) * 100, 2), round(float(best_sq)*100, 2), round(float(best_rq)*100, 2)
    print('Max coco                : COCO     {}'.format(np.max(best_coco_lst)))
    print('Best epoch              : index    {}'.format(best_param))
    print('Best COCO parm          : PQ       {}, RQ   {}, SQ {}'.format(best_coco, best_sq, best_rq))
    print('Best attribute setting  : Dynamics {}, Area {}'.format(int(best_dynamics), int(best_area)))

    # Save information
    dict_1 = {'best_coco_lst': best_coco_lst, 'full_coco_lst': full_cooo_lst}
    res_dir = Path(args.image_dir).parent
    dict_save_path = os.path.join(res_dir, 'bench_coco_lst.npy')
    np.save(dict_save_path, dict_1)
    print('Benchmark saving into: {}'.format(dict_save_path))
