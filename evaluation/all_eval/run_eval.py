import numpy as np

from evaluation.all_eval.pixel_eval.p_eval import corr_comp_qual, clDice
from evaluation.all_eval.topo_eval.t_eval import eval_betti

def evaluation(pred, gt, pred_ws):
    print('Pixel evaluation.')
    # Calculate correct, complete and quality
    gt_bool   = (gt > 0.5).astype(np.bool)
    pred_bool = (pred > 0.5).astype(np.bool)
    corr, comp, qual, TP_g, TP_p, FN, FP = corr_comp_qual(gt_bool, pred_bool, slack=8)
    print('Correct, complete and quality: {}, {}, {}'.format(round(corr*100, 2), round(comp*100, 2), round(qual*100, 2)))

    # Calculate cldice
    score_clDice = clDice(pred_bool, gt_bool)
    print('Cldice                       : {}'.format(round(score_clDice*100, 2)))

    print('Topology evaluation.')
    norm_betti_err_pred, norm_betti_err_gt = eval_betti(pred_ws, gt)
    norm_betti_err_f1 = (2 * norm_betti_err_pred * norm_betti_err_gt) / (norm_betti_err_pred + norm_betti_err_gt)
    print('Betti-error                  : pre. {}, rec. {}, f1: {} '.format(round(norm_betti_err_pred, 2), round(norm_betti_err_gt, 2), round(norm_betti_err_f1, 2)))

    return round(corr*100, 2), round(comp*100, 2), round(qual*100, 2), round(score_clDice*100, 2), round(norm_betti_err_pred, 2), round(norm_betti_err_gt, 2), round(norm_betti_err_f1, 2)
