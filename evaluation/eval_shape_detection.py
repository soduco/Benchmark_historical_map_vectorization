#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import logging
import json
import numpy as np
import sys
from PIL import Image

from evaluation import evaltk


IOU_THRESHOLD_DEFAULT = 0.5

logging.basicConfig(level=logging.INFO)

BG_LABEL = 0

def coco_panoptic_metrics(df, iou_threshold = 0.5):
    idx = df.index.searchsorted(iou_threshold)
    iou_sum = np.sum(df.index[idx:])
    tp = df["True Positives"].iloc[idx]
    fp = df["False Positives"].iloc[idx]
    fn = df["False Negatives"].iloc[idx]
    pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn)
    sq = iou_sum / tp if tp != 0 else 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return pq, sq, rq

def print_scores_summary(iou_ref, iou_contender, df, iou_threshold = 0.5, file = None):
    nlabel_gt = iou_ref.size
    nlabel_contender = iou_contender.size
    if file == None:
        file = sys.stdout

    idx = df.index.searchsorted(iou_threshold)
    print(f"Number of labels in GT: {nlabel_gt}", file = file)
    print(f"Number of labels in contender: {nlabel_contender}", file = file)

    pq, sq, rq = coco_panoptic_metrics(df, iou_threshold)
    print(f"COCO Panoptic eval: PQ: {pq:.5f} = SQ {sq:.5f} + RQ {rq:.5f}", file=file)

    THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
    idx = df.index.searchsorted(THRESHOLDS)

    try:
        subset = df.iloc[idx]
    except IndexError as error:
        idx[-1] = -1
        subset = df.iloc[idx]

    subset.index = THRESHOLDS
    subset.index.name = "IoU"
    print(subset.round(2), file = file)
    return subset


def shape_detection(input_gt_path, input_contenders_path, output_dir, iou_threshold, input_mask, save_fig=True):
    if not (0.5 <= iou_threshold < 1.0):
        raise ValueError(f"iou_threshold parameter must be >= 0.5 and < 1.")

    # Load input images
    ref = np.array(Image.open(input_gt_path))
    if ref is None:
        raise ValueError(f"input file {input_gt_path} cannot be read.")

    # Load mask image
    msk_bg = None
    if input_mask:
        msk_bg = np.array(Image.open(input_mask))
        if msk_bg is None:
            raise ValueError(f"mask file {input_mask} cannot be read.")
        if msk_bg.shape != ref.shape:
            raise ValueError("GT and MASK image do not have the same shapes: {} vs {}", ref.shape, msk_bg.shape)
        # Create boolean mask
        msk_bg = msk_bg==0

    # Mask input image if needed
    if msk_bg is not None:
        ref = evaltk.mask_label_image(ref, msk_bg, bg_label=BG_LABEL)

    contenders = []
    for p in input_contenders_path:
        p = Path(p)
        contender = np.array(Image.open(str(p)))

        if contender is None:
            raise ValueError(f"input file {p} cannot be read.")

        if contender.shape != ref.shape:
            raise ValueError("GT and PRED label maps do not have the same shapes: {} vs {}", ref.shape, contender.shape)

        # Mask predicted image if needed
        if msk_bg is not None:
            contender = evaltk.mask_label_image(contender, msk_bg, bg_label=BG_LABEL)

        contenders.append((str(p.stem), contender))

    # Create output dir early
    os.makedirs(output_dir, exist_ok=True)

    odir = Path(output_dir)
    recalls = []
    precisions = []
    # coco_metrics - structure: dict[str, dict(str, float)]
    # contender_name -> {"pq": pq_float_value, "sq": sq_float_value, "rq": rq_float_value}
    coco_metrics = {}  
    for name, contender_img in contenders:
        logging.info("Processing: %s", name)
        (recall, precision), hist = evaltk.iou(ref, contender_img)
        if save_fig:
            evaltk.viz_iou(ref, recall, Path(odir, "viz_recall_{}.jpg".format(name)))
            evaltk.viz_iou(contender_img, precision, Path(odir, "viz_precision_{}.jpg".format(name)))

        df = evaltk.compute_matching_scores(recall, precision)
        if save_fig:
            df.to_csv(Path(odir, f"{name}_figure.csv"))

        pq, sq, rq = coco_panoptic_metrics(df, iou_threshold)
        coco_metrics_current = {"PQ": pq, "SQ": sq, "RQ": rq}
        if save_fig:
            with open(Path(odir, f"{name}_coco_panoptic_metrics.json"), "w") as f:
                json.dump(coco_metrics_current, f)
        coco_metrics[name] = coco_metrics_current

        with open(Path(odir, f"{name}_summary.txt"), "w") as f:
            print_scores_summary(recall, precision, df)
            subset = print_scores_summary(recall, precision, df, file=f)

        if save_fig:
            evaltk.plot_scores(df, out = Path(odir, f"{name}_figure.pdf"))
        recalls.append(recall)
        precisions.append(precision)


    if len(contenders) == 2:
        A_recall, B_recall = recalls
        A_precision, B_precision = precisions
        (A_name, A), (B_name, B) = contenders
        A_recall_map = A_recall[ref]
        A_precision_map = A_precision[A]
        B_recall_map = B_recall[ref]
        B_precision_map = B_precision[B]
        evaltk.diff(A_recall_map, B_recall_map, out_path=Path(odir, "compare_recall.png"))
        evaltk.diff(A_precision_map, B_precision_map, out_path=Path(odir, "compare_precision.png"))

    print("All done.")

    return (
        np.array(subset['Precision']).tolist(),
        np.array(subset['Recall']).tolist(),
        np.array(subset['F-score']).tolist(),
        (recall, precision, hist),
        coco_metrics)


def main():
    parser = argparse.ArgumentParser(description='Evaluate the detection of shapes.')
    parser.add_argument('input_gt_path', help='Path to the input label map (TIFF 16 bits) for ground truth.')
    parser.add_argument('input_contenders_path', help='Path to the contenders label map (TIFF 16 bits) for predictions.', nargs='+')
    parser.add_argument('-m', '--input-mask', help='Path to an mask image (pixel with value 0 will be discarded in the evaluation).')
    parser.add_argument('-o', '--output-dir', help='Path to the output directory where results will be stored.')
    parser.add_argument('--iou-threshold', type=float, help='Threshold value (float) for IoU: 0.5 <= t < 1.'
                        f' Default={IOU_THRESHOLD_DEFAULT}', default=IOU_THRESHOLD_DEFAULT)

    args = parser.parse_args()
    shape_detection(args.input_gt_path, args.input_contenders_path, args.input_mask, args.output_dir, args.iou_threshold)

if __name__ == "__main__":
    main()
