# Run Benchmark cc filtering
import numpy as np
import argparse
import time
import os
import cv2
import skimage.morphology
from pathlib import Path

from evaluation.eval_shape_detection import shape_detection


def morpho_filter(image):
    # Binarize and doing morphology filtering
    image = (image>0.5)
    image = skimage.morphology.remove_small_objects(image)
    image = skimage.morphology.remove_small_holes(image)
    image = (image*255).astype(np.uint8)
    return image

def tif2png(file_dir):
    # Filter mode represent the filtering type: morphology based or persistent ws
    save_dir = Path(file_dir).parent
    save_dir = os.path.join(str(save_dir), 'reconstruction_png_morpho_filter')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        file_names = os.listdir(file_dir)
        for file in file_names:
            image = cv2.imread(os.path.join(file_dir, file), 0)
            image = image / 255.
            image = morpho_filter(image)

            new_file = os.path.join(save_dir, file.split('.')[0]+'.png')
            cv2.imwrite(new_file, image)
    else:
        print('Folder already created.')
    return save_dir

def meyer_watershed(image_path, dynamic, area, output_path, out_visu_path):
    print('./histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))
    os.system('./histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))

def save_label_maps(labels, output_path, model, mode, file_name, attribute, name):
    output_dir = os.path.join(output_path, model, mode, attribute, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    if not cv2.imwrite(output_path, labels.astype(np.uint16)):
        raise Exception("Could not write image")

def pre_save_label_maps(image_path, output_path, model, mode):
    def save_label_maps_batch(image_path, output_path, parameters):
        dynamic, area = parameters
        file_name = image_path.split('/')[-1].split('.')[0]+'_'+str(dynamic)+'_'+str(area)+'.png'
        name = image_path.split('/')[-1].split('.')[0]
        output_dir = os.path.join(output_path, model, mode, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        meyer_watershed(image_path, dynamic, area, output_path, './out.png')

    parameters = (0, 0)
    save_label_maps_batch(image_path, output_path, parameters)

def eval(gt_path, input_contenders_dir, output_dir, iou_threshold, input_mask, model, mode, name):
    def coco_eval_batch(input_contenders_path_name):
        output_val_dir = os.path.join(output_dir, model, mode, name, input_contenders_path_name.split('.')[0])

        if not os.path.exists(output_val_dir):
            os.makedirs(output_val_dir)
        input_contenders_path = os.path.join(input_contenders_dir, input_contenders_path_name)
        input_contenders_path = [input_contenders_path]
        _, _, _, _, coco_matrix = shape_detection(gt_path, input_contenders_path, output_val_dir, iou_threshold, input_mask, save_fig=False)
        dynamics, area = input_contenders_path_name.split('.')[0].split('_')[-2], input_contenders_path_name.split('.')[0].split('_')[-1]
        return [coco_matrix[input_contenders_path_name.split('.')[0]]['PQ'], coco_matrix[input_contenders_path_name.split('.')[0]]['RQ'], coco_matrix[input_contenders_path_name.split('.')[0]]['SQ']]

    input_contenders_dir = os.path.join(input_contenders_dir, model, mode, name)
    file_name = os.listdir(input_contenders_dir)[0]
    coco_list = coco_eval_batch(file_name)
    coco_list = np.array(coco_list)
    return coco_list


if __name__ == '__main__':
    def parse_args():
        AUC_THRESHOLD_DEFAULT = 0.5
        parser = argparse.ArgumentParser(description='Benchmark evaluaion')

        ABS_PATH = Path(parser.parse_args().image_dir).parent
        parser.add_argument('--output_path', type=str, default=r'{}/label_maps_cc_labelling/'.format(ABS_PATH),
            help='The output path of the image')
        parser.add_argument('--input_contenders_dir', type=str, default=r'{}/label_maps_cc_labelling/'.format(ABS_PATH),
            help='Save contender path')
        parser.add_argument('--output_dir', type=str, default=r'{}/eval_folder_cc_labelling/'.format(ABS_PATH),
            help='Save evaluation path')
        parser.add_argument('--auc-threshold', type=float,
            help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)

        parser.add_argument('--image_dir', type=str, default=r'/lrde/work/ychen/PRL/benchmark_DL/unet_original/HistoricalMap2020/UNET/2022-03-03_20:15:02_lr_0.0001_train_unet_orign_bs_1/reconstruction',
            help='The input path of the image')
        parser.add_argument('--gt_path', type=str, default=r'/lrde/work/ychen/PRL/benchmark_mws/gt_label_path/gt_label_map.png',
            help='The ground truth of the label path')
        parser.add_argument('--validation_mask', type=str, default=r'/lrde/image/CV_2021_yizi/historical_map_2020/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-MASK_content.png',
            help='Validation mask to evaluate the results')
        parser.add_argument('--gt_label_path', type=str, default='/lrde/work/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png',
            help='the gt label path')
        return parser.parse_args()
    print('##########################  GRID SEARCH VALIDATION  ##################################')
    start = time.time()
    args = parse_args()
    mode = 'val'
    model = (args.image_dir).split('/')[-2]

    save_dir = tif2png(args.image_dir)

    image_lst = os.listdir(save_dir)
    best_parm_lst = []
    best_coco_lst = []
    full_cooo_lst = []
    for l in image_lst:
        image_path = os.path.join(save_dir, l)
        name = image_path.split('/')[-1].split('.')[0]
        pre_save_label_maps(image_path, args.output_path, model, mode)
        args.gt_path, args.input_contenders_dir, args.output_dir, args.auc_threshold, args.validation_mask, model, mode, name
        coco_list = eval(args.gt_path, args.input_contenders_dir, args.output_dir, args.auc_threshold, args.validation_mask, model, mode, name)
        full_cooo_lst.append(coco_list)

    best_coco_lst = np.array(full_cooo_lst)
    best_param = np.argmax(best_coco_lst[:,0])
    best_coco_array = full_cooo_lst[best_param]
    best_coco, best_sq, best_rq = best_coco_lst[best_param]
    best_coco, best_sq, best_rq = round(float(best_coco) * 100, 2), round(float(best_sq)*100, 2), round(float(best_rq)*100, 2)
    print('Max coco                : COCO     {}'.format(np.max(best_coco_lst[:,0])))
    print('Best epoch              : index    {}'.format(best_param))
    print('Best COCO parm          : PQ       {}, RQ {}, SQ {}'.format(best_coco, best_rq, best_sq))

    # Save information
    dict_1 = {'best_coco_lst': best_coco_lst, 'full_coco_lst': full_cooo_lst}
    res_dir = Path(args.image_dir).parent
    dict_save_path = os.path.join(res_dir, 'bench_coco_lst_cc.npy')
    np.save(dict_save_path, dict_1)
    print('Benchmark saving into: {}'.format(dict_save_path))
