import numpy as np
from skimage.util import view_as_windows
from PIL import Image
import argparse

import pdb


def generate_tiling(image_path, w_size):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))
    if len(in_img.shape) == 2:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    else:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            if len(in_img.shape) == 2:
                tt = tiles[row, col, ...].copy()
            else:
                tt = tiles[row, col, 0, ...].copy()
            # If you want black boarder, set the value to 25 (Suggest not using balck boarder)
            # bordersize=1005
            # tt[:bordersize,:, 2] = 255
            # tt[-bordersize:,:, 2] = 255
            # tt[:,:bordersize, 2] = 255
            # tt[:,-bordersize:, 2] = 255
            # skio.imsave(os.path.join(save_image_path, f"t_r{row:02d}_c{col:02d}.jpg"), tt)
            tiles_lst.append(tt)
    return tiles_lst

def main():
    parser = argparse.ArgumentParser(description='Create Tillings.')
    parser.add_argument('input_path', help='Path of original image.')
    parser.add_argument('gt_path', help='Path of ground truth.')
    parser.add_argument('save_image_path', help='Directory of images for saving the tilings.')
    parser.add_argument('save_gt_path', help='Directory of ground truth for saving the tilings.')
    parser.add_argument('width', help='Width of the tillings')

    args = parser.parse_args()
    generate_tiling(args.input_path, args.gt_path, args.save_image_path, args.save_gt_path, args.width)


if __name__ == '__main__':
    main()
