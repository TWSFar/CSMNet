import os
import cv2
import pickle
import h5py
import numpy as np
import os.path as osp
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataloaders import deeplab_transforms as dtf


class DroneCC(Dataset):
    def __init__(self, opt, mode):
        self.data_dir = opt.root_dir
        self.mode = mode
        self.labels_dir = osp.join(self.data_dir, 'SegmentationClass')
        self.list_file = osp.join(self.data_dir, "{}list.txt".format(mode))
        self.img_root = osp.join(self.data_dir, "sequences")
        self.anno_dir = osp.join(self.data_dir, "annotations")
        self.cache_path = self._cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.img_list = self._load_imgs_idx()  # order: 1
        self.samples = self._load_samples()  # order: 2

        # transform
        if self.mode == "train":
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=opt.input_size),
                # dtf.RandomFilter(),
                dtf.RandomColorJeter(0.3, 0.3, 0.3, 0.3),
                dtf.RandomHorizontalFlip(),
                dtf.Normalize(**opt.norm_cfg),
                dtf.ToTensor()])
        else:
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=opt.input_size),  # 513
                # dtf.RandomFilter(),
                dtf.Normalize(**opt.norm_cfg),
                dtf.ToTensor()])

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = sample["image"]
        dir_name, img_name = img_path.split('/')[-2:]
        label_path = osp.join(self.labels_dir, dir_name+'_'+img_name[:-4]+'.hdf5')
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(label_path), '{} not exist'.format(label_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        with h5py.File(label_path, 'r') as hf:
            label = np.array(hf['label'])

        o_h, o_w = img.shape[:2]

        sample = {"image": img, "label": label}
        sample = self.transform(sample)

        scale = torch.tensor([sample["image"].shape[1] / o_h,
                              sample["image"].shape[0] / o_w])
        sample["scale"] = scale
        sample["path"] = img_path
        return sample

    def _cre_cache_path(self, data_dir):
        cache_path = osp.join(data_dir, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def _load_gts(self):
        gts = {}
        for dir_id in self.dir_ids:
            with open(osp.join(self.anno_dir, dir_id+'.txt')) as f:
                for line in f.readlines():
                    frame, x, y = line.split(',')
                    frame = frame.strip().zfill(5)
                    key = dir_id + '_' + frame+'.jpg'
                    if key in gts:
                        gts[key].append([int(x.strip()), int(y.strip())])
                    else:
                        gts[key] = [[int(x), int(y)]]
        return gts

    def _load_imgs_idx(self):
        self.dir_ids = []
        img_list = []
        with open(self.list_file, 'r') as f:
            for line in f.readlines():
                self.dir_ids.append(line.strip())
        for dir in self.dir_ids:
            for img in os.listdir(osp.join(self.img_root, dir)):
                img_list.append(osp.join(self.img_root, dir, img))

        return img_list

    def _load_samples(self):
        cache_file = self.cache_file

        # load bbox and save to cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('{} gt samples loaded from {}'.
                  format(self.mode, cache_file))
            return samples

        # load information of image and save to cache
        gts = self._load_gts()
        samples = []
        for img_path in self.img_list:
            size = Image.open(img_path).size
            img_path = img_path.replace('\\', '/')
            dir_name, img_name = img_path.split('/')[-2:]
            coordinate = gts[dir_name+"_"+img_name]
            samples.append({
                "image": img_path,
                "width": size[0],
                "height": size[1],
                "coordinate": coordinate
            })

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples

    def __len__(self):
        return len(self.img_list)


def show_image(img, coordinate):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    plt.imshow(img[..., ::-1])
    for local in coordinate:
        x, y = local
        plt.scatter(x, y, c='red', s=40, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    dataset = DroneCC('G:\\CV\\Dataset\\CC\\Visdrone\\VisDrone2020-CC', 'train')
    for i in range(1, 1000, 20):
        sample = dataset.samples[i]
        img_path = sample['image']
        img = cv2.imread(img_path)
        coordinate = sample['coordinate']
        show_image(img, coordinate)

        pass
