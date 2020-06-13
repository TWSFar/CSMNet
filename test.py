import os
import cv2
import h5py
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.csm_dronecc import opt
from models import CSMNet as Model
from dataloaders import deeplab_transforms as dtf

import torch
from torchvision import transforms
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--chekpoint', type=str, default="/home/twsf/work/CSMNet/run/dronecc/20200611_23_train/last.pth.tar")
    parser.add_argument('--img_root', type=str, default="/home/twsf/data/DroneCC/sequences")
    parser.add_argument('--img_list', type=str, default="/home/twsf/data/DroneCC/testlist.txt")
    parser.add_argument('--results_dir', type=str, default="./results")
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()
    return args


args = parse_args()
opt._parse({})


def test():
    input_w, input_h = opt.input_size
    mask_size = (int(input_h / 16), int(input_w / 16))
    if osp.exists(args.results_dir):
        os.remove(args.results_dir)
    os.makedirs(args.results_dir)

    # data
    imgs_path = []
    with open(args.img_list, 'r') as f:
        for dir_name in f.readlines():
            img_dir = osp.join(args.img_root, dir_name.strip())
            for img_name in os.listdir(img_dir):
                imgs_path.append(osp.join(img_dir, img_name))

    transform = transforms.Compose([
        dtf.FixedNoMaskResize(size=opt.input_size),  # 513
        dtf.Normalize(**opt.norm_cfg),
        dtf.ToTensor()])

    # model
    model = Model(opt).to(opt.device)

    # resume
    if osp.isfile(args.chekpoint):
        print("=> loading checkpoint '{}'".format(args.chekpoint))
        checkpoint = torch.load(args.chekpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.chekpoint, checkpoint['epoch']))
    else:
        raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for ii, img_path in enumerate(tqdm(imgs_path)):
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
            sample = {"image": img, "label": None}
            sample = transform(sample)

            # predict
            region_pred, density_pred = model(sample['image'].unsqueeze(0).to(opt.device))

            # region_pred = np.argmax(region_pred.cpu().numpy(), axis=1).reshape(mask_size)
            density_pred = torch.clamp(density_pred, min=0.0).cpu().numpy().reshape(mask_size)
            pred = density_pred * opt.norm_cfg['para']

            dir_name, img_name = img_path.split('/')[-2:]
            file_name = osp.join(args.results_dir, dir_name+".txt")
            with open(file_name, 'a') as f:
                f.writelines("{},{}\n".format(int(img_name[:-4].strip()), pred.sum()))

            if args.show:
                plt.figure()
                plt.subplot(2, 1, 1).imshow(img)
                plt.subplot(2, 1, 2).imshow(pred)
                plt.show()


if __name__ == '__main__':
    test()
