import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, skeletonize_3d

def correctness(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps) # precision

def completeness(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps) # recall

def quality(corr, comp, eps=1e-12):
    return (comp*corr)/(comp-comp*corr+corr+eps)
    
def f1(corr, comp, eps=1e-12):
    return 2.0/(1.0/(corr+eps) + 1.0/(comp+eps))

def relaxed_confusion_matrix(pred_s, gt_s, slack=8):
    distances_gt = ndimage.distance_transform_edt((np.logical_not(gt_s)))
    distances_pred = ndimage.distance_transform_edt((np.logical_not(pred_s)))
    
    true_pos_area_gt = distances_gt<=slack
    false_pos_area = distances_gt>slack 
    
    true_pos_area_pred = distances_pred<=slack
    false_neg_area = distances_pred>slack
    
    true_positives_gt = np.logical_and(true_pos_area_gt, pred_s).sum() # length of the matched extraction
    false_positives = np.logical_and(false_pos_area, pred_s).sum() 
    
    true_positives_pred = np.logical_and(true_pos_area_pred, gt_s).sum() # length of the matched reference
    false_negatives = np.logical_and(false_neg_area, gt_s).sum()

    return true_positives_gt, true_positives_pred, false_negatives, false_positives


def compute_scores(TP_g, TP_p, FN, FP, eps=1e-12):
    corr = correctness(TP_g, FP, eps)
    comp = completeness(TP_p, FN, eps)
    qual = quality(corr, comp)    
    return corr, comp, qual


def corr_comp_qual(gt_s, pred_s, slack=8):
    """
    Correctness Completeness Quality metric
    
    C. Wiedemann, C. Heipke, H. Mayer, and O. Jamet. Empirical
    Evaluation of Automatically Extracted Road Axes.
    Empirical Evaluation Techniques in Computer Vision
    
    Parameters
    ----------
    gt_s : np.bool_ (H,W)
        skeletonized ground-truth image
    pred_s : np.bool_ (H,W)
        skeletonized predicted image  
    slack : int
        the margin on each side of the lines
        
    Return
    ------
    Correctness, completenes and quality for this sample
    and the quantities for the aggregation.
    """
    
    if np.sum(gt_s)==0:
        raise ValueError("Ground-truth skeleton is empty!")
    if np.sum(pred_s)==0:
        print("!! Predicted skeleton is empty !!")
    
    TP_g, TP_p, FN, FP = relaxed_confusion_matrix(pred_s, gt_s, slack)
    
    corr, comp, qual = compute_scores(TP_g, TP_p, FN, FP)

    return corr, comp, qual, TP_g, TP_p, FN, FP


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

