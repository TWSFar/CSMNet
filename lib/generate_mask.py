"""convert VOC format
+ density_voc
    + JPEGImages
    + SegmentationClass
"""

import os
import cv2
import h5py
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from dataset import DroneCC
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--mode', type=str, default=['train', 'val'],
                        nargs='+', help='for train or val')
    parser.add_argument('--db_root', type=str,
                        # default=user_dir+"/data/DroneCC/",
                        default="G:\\CV\\Dataset\\CC\\Visdrone\\VisDrone2020-CC",
                        help="dataset's root path")
    parser.add_argument('--method', type=str, default='default',
                        choices=['centerness', 'gauss', 'default'])
    parser.add_argument('--maximum', type=int, default=999,
                        help="maximum of mask")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and region mask")
    args = parser.parse_args()
    args.mask_size = [30, 40]

    return args


def show_image(img, mask):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1).imshow(img)
    plt.subplot(2, 1, 2).imshow(mask, cmap=cm.jet)
    plt.savefig('mask.jpg')
    # plt.show()


def _centerness_pattern(width, height):
    """ Follow @FCOS
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    yv, xv = np.meshgrid(np.arange(0, height), np.arange(0, width))
    for yi, xi in zip(yv.ravel(), xv.ravel()):
        right = width - xi - 1
        bottom = height - yi - 1
        min_tb = min(yi+1, bottom)
        max_tb = max(yi+1, bottom)
        min_lr = min(xi+1, right)
        max_lr = max(xi+1, right)
        centerness = np.sqrt(1.0 * min_lr * min_tb / (max_lr * max_tb))
        pattern[yi, xi] = centerness

    return pattern


def gaussian_pattern(width, height):
    """在3倍的gamma距离内的和高斯的97%
    """
    cx = int(round((width-0.01)/2))
    cy = int(round((height-0.01)/2))
    pattern = np.zeros((height, width), dtype=np.float32)
    pattern[cy, cx] = 1
    gamma = [0.15*height, 0.15*width]
    pattern = gaussian_filter(pattern, gamma)

    return pattern


def _generate_mask(sample, mask_scale=(30, 40)):
    try:
        height, width = sample["height"], sample["width"]
        mask_h, mask_w = mask_scale
        density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)
        for x, y in sample["coordinate"]:
            y = min(int(np.round(y / height * mask_h)), mask_h-1)
            x = min(int(np.round(x / width * mask_w)), mask_w-1)
            if args.method == 'default':
                density_mask[y, x] += 1
            elif args.method == 'gauss':
                density_mask[y, x] += gaussian_pattern(y, x)
            elif args.method == 'centerness':
                density_mask[y, x] += _centerness_pattern(y, x)

        # return density_mask.clip(min=0, max=args.maximum)
        return density_mask

    except Exception as e:
        print(e)
        print(sample["image"])


if __name__ == "__main__":
    args = parse_args()

    for split in args.mode:
        dataset = DroneCC(args.db_root, split)
        mask_dir = dataset.data_dir + '/SegmentationClass'
        if not osp.exists(mask_dir):
            os.mkdir(mask_dir)
        samples = dataset.samples

        print('generate {} masks...'.format(split))
        for sample in tqdm(samples):
            density_mask = _generate_mask(sample, args.mask_size)
            img_dir, img_name = sample['image'].split('/')[-2:]
            maskname = osp.join(mask_dir, img_dir+'_'+img_name[:-4]+'.hdf5')
            with h5py.File(maskname, 'w') as hf:
                hf['label'] = density_mask

            if args.show:
                img = cv2.imread(sample['image'])
                show_image(img, density_mask)

        print('done.')
