import numpy as np
import matplotlib.cm
import skimage.io as skio
import logging

'''
Create a color map that shows the difference between two contenders

The inputs A and B should be normalized between 0 and 1.
It outputs a new image where comuting B - A (normalized between [-1,1]) and saves it in `out`

'''
def diff(A: np.ndarray, B: np.ndarray, out_path: str = None):
    C = (B - A)


    cmap = matplotlib.cm.ScalarMappable(norm=None, cmap="RdYlGn")
    out = cmap.to_rgba(C / 2. + 0.5, norm=False, bytes=True)
    if out_path:
        logging.info("Saving image in %s", out_path)
        skio.imsave(out_path, out)
    return C
