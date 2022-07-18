import numpy as np
import cv2
import argparse


def reconstruct_tiling(original_image_path, test_pred_dict, tile_save_path, w_size, image_debug=None, save_image=True):
    in_patches = list(test_pred_dict.keys())
    in_patches.sort()
    patches_images =  list(test_pred_dict.values())

    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)

    if save_image:
        cv2.imwrite(tile_save_path, new_img)

    if image_debug is not None:
        cv2.imwrite(image_debug, new_img * 255)
    return new_img


def reconstruct_tiling_array(original_image_path, patches_images, w_size):
    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)
    return new_img


def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype):
    '''Adjust to take patch images directly.
    patch_size is the size of the tiles
    step_size should be patch_size//2
    image_size_2d is the size of the original image
    image_dtype is the data type of the target image

    Most of this could be guessed using an array of patches
    (except step_size but, again, it should be should be patch_size//2)
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    if len(patches_images.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    patch_offset = step_size//2
    patch_inner = p_h-step_size
    for row in range(numrows):
        for col in range(numcols):
            tt = patches_images[row*numcols+col]
            tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset]
            img[row*step_size:row*step_size+patch_inner,
                col*step_size:col*step_size+patch_inner] = tt_roi # +1?? 
    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]
