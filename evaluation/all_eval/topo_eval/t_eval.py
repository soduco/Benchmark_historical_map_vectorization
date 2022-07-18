from skimage.measure import label
import numpy as np

def eval_betti(pred_ws, gt):
    norm_betti_err_pred = np.abs(len(np.unique(label(gt)))-len(np.unique(pred_ws)))/len(np.unique(pred_ws))
    norm_betti_err_gt = np.abs(len(np.unique(label(gt)))-len(np.unique(pred_ws)))/len(np.unique(label(gt)))
    return norm_betti_err_pred, norm_betti_err_gt
