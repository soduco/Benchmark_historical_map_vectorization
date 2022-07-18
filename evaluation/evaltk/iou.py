import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import skimage.io as skio
import logging
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc

'''
Compute the intersection over union of two label images


Args:

:param A: Input WxH array of labels (of N labels)
:param B: Input WxH array of labels (of M labels)
:return: A pair (X, Y) with IoU values of best match 
         X: A vs B (array of size N)
         Y: A vs B (array of size M)
'''
def iou(A: np.ndarray, B: np.ndarray):
    assert A.shape == B.shape

    # Convert inputs to uint32 so we can add them.
    labels_gt = A.astype(np.uint32)
    labels_pred = B.astype(np.uint32)

    # Count the number of components in each image.
    nlabel_gt = labels_gt.max() + 1
    nlabel_pred = labels_pred.max() + 1
    sz = nlabel_gt * nlabel_pred

    # (Here comes the beautiful part.)
    # Add the images. We have |GT|*|PRED| possible values.
    # Thanks to the multiplication, each pair of components will have its new id.
    H = labels_gt * nlabel_pred + labels_pred
    # Compute the number of pixels for each pair of components.
    # The histogram is sorted byt values.
    hist = np.bincount(H.ravel(), minlength=sz)
    # Because the values are sorted in an order we chose, we can reshape the histogram.
    # `hist[i,j]` contains the area of the intersection between components GT_i and PRED_j
    hist = hist.reshape((nlabel_gt, nlabel_pred))

    # Now we have the intersections, we can computes total areas and IoUs.
    # This works because we have (by construction) disjoint components.

    # Areas of components
    areas_gt = hist.sum(axis=1)
    areas_pred = hist.sum(axis=0)

    # Edwin's extra optimization: compute only marginal IoUs
    best_match_gt = hist.argmax(axis=1)
    best_match_pred = hist.argmax(axis=0)

    area_gt_Inter_BestMatch = hist[np.arange(nlabel_gt), best_match_gt]
    area_pred_Inter_BestMatch = hist[best_match_pred, np.arange(nlabel_pred)]
    area_gt_U_BestMatch = areas_gt + areas_pred[best_match_gt] - area_gt_Inter_BestMatch
    area_pred_U_BestMatch = areas_pred + areas_gt[best_match_pred] - area_pred_Inter_BestMatch

    IoU_gt = area_gt_Inter_BestMatch / area_gt_U_BestMatch
    IoU_pred = area_pred_Inter_BestMatch / area_pred_U_BestMatch

    return (IoU_gt, IoU_pred), hist

'''
Visualization of the IoU (outputs in a file if ``output`` is not NULL)


Args:

:param A: Input W*H array of labels (of M labels)
:param iou: Input array of M iou values
'''

# Viz
# Reuse matplotlib colormaps


def viz_iou(A: np.ndarray, iou: np.ndarray, output_path: str = None):
    cmap = matplotlib.cm.get_cmap(name="RdYlGn")
    lut = cmap(iou, bytes=True)[...,:3]
    lut[0] = (0,0,0)
    out = lut[A]
    if output_path:
        logging.info("Saving image in %s", output_path)
        skio.imsave(output_path, out)
        logging.info("end saving")
    else:
        plt.imshow(out)




'''
Compute the F-score, Precision and Recall Scores from IoU scores measured of 2 images components

The match between two components A and B is considered when the IoU > 0.5. It returns a dataframe with:

          Precision    Recall   F-score
IoU
0.500665   0.757566  0.434935  0.552607
0.502368   0.757244  0.434750  0.552372
0.505121   0.756600  0.434381  0.551902
0.506959   0.756278  0.434196  0.551667
...
0.995850   0.000966  0.000555  0.000705
0.995986   0.000644  0.000370  0.000470
0.996028   0.000322  0.000185  0.000235
'''
def compute_matching_scores(ref: np.ndarray, containder_score: np.ndarray):
    scores_A = np.sort(ref)
    scores_B = np.sort(containder_score)
    startA = np.searchsorted(scores_A, 0.5, side="right")
    startB = np.searchsorted(scores_B, 0.5, side="right")
    nonMatchA = startA # Number of A points with no-match
    nonMatchB = startB # Number of B points with no-match
    scores_A1 = scores_A[startA:]
    scores_B1 = scores_B[startB:]
    assert(np.allclose(scores_A1, scores_B1))  # We must have a partial bijection between A and B for IoU > 0.5


    # Recall = Nombre de matchs / size(ref)
    # Precision = Nombre de matchs / size(containder)
    rem = scores_A1.size
    matches = np.arange(rem, 0, -1)
    recall = matches / float(scores_A.size)
    precision = matches / float(scores_B.size)
    # FIXME should compute fscore using TP/FP/FN to avoid undefined precision when scores_B.size == 0
    fscore = 2 * precision * recall / (precision + recall)

    # TP/TN/FN values
    tp = matches
    fp = nonMatchB + scores_B1.size - matches
    fn = nonMatchA + scores_A1.size - matches

    # # Coco metrics -- continuous version 
    # # what is the meaning of this? => value of coco indicators if we have a stricter threshold than 0.5 IoU
    # iou = np.cumsum(scores_A1[::-1])[::-1]
    # pq_iou = iou / (tp + 0.5 * fp + 0.5 * fn)
    # # sq = avg(iou[iou>threshold])
    # sq_iou = np.where(tp != 0, iou / tp, 0)  # iou / tp if tp != 0 else 0
    # # rq = fscore
    # rq_iou = tp / (tp + 0.5 * fp + 0.5 * fn)
    # # pq_iou = sq_iou * rq_iou  # same values

    df = pd.DataFrame({
        "IoU" : scores_A1,
        "Precision" : precision,
        "Recall" : recall,
        "F-score" : fscore,
        "True Positives": tp,
        "False Positives": fp,  #  useless in practice : depends on the size of B
        "False Negatives": fn,  #  useless in practice : depends on the size of A
        # "COCO_PQ": pq_iou,
        # "COCO_SQ": sq_iou,
        # "COCO_RQ": rq_iou
    })
    df = df.set_index("IoU")
    return df






'''
Dataframe

'''
def plot_scores(df, out=None, ax=None):
    sns.set()
    df_sel = df[["Precision", "Recall", "F-score"]] #, "COCO_PQ", "COCO_SQ", "COCO_RQ"]]

    if ax == None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()

    df_sel.plot(ax=ax)
    plt.xlim(0.5, 1)
    plt.ylim(0, 1)
    if out:
        logging.info("Saving plot %s", out)
        plt.savefig(out, dpi=300)
    else:
        plt.show()



'''
Mask an input label image and rearrange the label numbers so that
they form the continuous range [0, numlabel].

Note that there is not relabelling of the components, just a
renumbering.

Args:

:param labels: Image of labels to be masked
:param bg_mask: Mask of the background image (must have the same shape as `labels`)
:param bg_label: Value of the label for masked (background) pixels

Returns:

labels_renumbered: Image where all pixels in the background (masked) 
    have label `bg_label and all other pixels have the label value
    renumbered so that the labels in the resulting image
    for the continuous range `[0, numlabel]` (where `numlabel` is
    the number of distinct labels different from `0`.)
'''
def mask_label_image(labels: np.ndarray, bg_mask: np.ndarray, bg_label: int = 0):
    if not labels.shape == bg_mask.shape:
        raise ValueError(f"Expected same shapes for `labels` and `bg_mask`, "
            f"but got labels.shape={labels.shape} and bg_mask.shape={bg_mask.shape}. "
            "Please check the input images match." )
    if bg_mask.dtype != np.bool:
        bg_mask = bg_mask.astype(np.bool)

    # Create a new masked label map
    labels_renumbered = labels.copy()
    labels_renumbered[bg_mask] = bg_label

    # Renumber the labels
    n_labels = np.max(labels) + 1
    counts = np.bincount(labels_renumbered.ravel(), minlength=n_labels)
    active_labels_mask = counts > 0
    active_labels_total = np.sum(active_labels_mask)
    lut = np.full_like(counts, bg_label)
    lut[active_labels_mask] = np.arange(active_labels_total)
    labels_renumbered = lut[labels_renumbered]

    return labels_renumbered




