import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from config import *


cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HustDataset_multi_gts(Dataset):
    def __init__(self, dataset_name, dataset_root_path, txt_root_path, gts_num:int):
        self.gts_num = gts_num
        self.dataset_name = dataset_name
        self.image_root = dataset_root_path

        train_fn = os.path.join(txt_root_path, 'TrainList.txt')
        test_fn = os.path.join(txt_root_path, 'TestList.txt')

        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gts_list, img1, h, w):
        ih, iw, _ = img0.shape

        if ih < h or iw < w:
            img0 = cv2.resize(img0, (w, h), interpolation = cv2.INTER_CUBIC)
            img1 = cv2.resize(img1, (w, h), interpolation = cv2.INTER_CUBIC)
            aug_gts_list = [cv2.resize(gts_list[i], (w, h), interpolation = cv2.INTER_CUBIC) for i in range(self.gts_num)]
        else:
            x = np.random.randint(0, ih - h + 1)
            y = np.random.randint(0, iw - w + 1)
            img0 = img0[x:x+h, y:y+w, :]
            img1 = img1[x:x+h, y:y+w, :]
            aug_gts_list = [gts_list[i][x:x+h, y:y+w, :] for i in range(self.gts_num)]

        return img0, aug_gts_list, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths_01 = [imgpath + '/frame_1.png', imgpath + '/frame_{}.png'.format(2 + self.gts_num)]
        imgpaths_gts = [imgpath + '/frame_{}_gt.png'.format(2 + i) for i in range(self.gts_num)]

        img0 = cv2.imread(imgpaths_01[0])
        img1 = cv2.imread(imgpaths_01[1])
        gts_list = [cv2.imread(imgpaths_gts[i]) for i in range(self.gts_num)]

        return img0, gts_list, img1

    def __getitem__(self, index):
        def debug_for_test_imgsz(img0, gts_list, img1):
            ih, iw, _ = img0.shape
            if not (ih % 8 == 0 and iw % 8 == 0):
                if ih < 256 and iw < 256:
                    w, h = 256, 256
                elif ih < 512 and iw < 512:
                    w, h = 512, 512
                elif ih < 640 and iw < 640:
                    w, h = 640, 640
                elif ih < 1024 and iw < 1024:
                    w, h = 1024, 1024
                elif ih < 1280 and iw < 1280:
                    w, h = 1280, 1280
                else:
                    w, h = ih, iw
                img0 = cv2.resize(img0, (w, h), interpolation = cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (w, h), interpolation = cv2.INTER_CUBIC)
                gts_list = [cv2.resize(gts_list[i], (w, h), interpolation = cv2.INTER_CUBIC) for i in range(self.gts_num)]
            return img0, gts_list, img1
        img0, gts_list, img1 = self.getimg(index)

        if 'train' in self.dataset_name:
            img0, gts_list, img1 = self.aug(img0, gts_list, img1, 320, 320)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gts_list = [gts_list[i][:, :, ::-1] for i in range(self.gts_num)]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gts_list = [gts_list[i][::-1] for i in range(self.gts_num)]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gts_list = [gts_list[i][:, ::-1] for i in range(self.gts_num)]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gts_list = [cv2.rotate(gts_list[i], cv2.ROTATE_90_CLOCKWISE) for i in range(self.gts_num)]
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gts_list = [cv2.rotate(gts_list[i], cv2.ROTATE_180) for i in range(self.gts_num)]
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gts_list = [cv2.rotate(gts_list[i], cv2.ROTATE_90_COUNTERCLOCKWISE) for i in range(self.gts_num)]
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img0, gts_list, img1 = debug_for_test_imgsz(img0, gts_list, img1)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gts_list = [torch.from_numpy(gts_list[i].copy()).permute(2, 0, 1) for i in range(self.gts_num)]

        retu_cat_part1 = torch.cat((img0, img1), 0)
        retu_cat_part2 = torch.cat(([gts_list[i] for i in range(self.gts_num)]), 0)

        return torch.cat((retu_cat_part1, retu_cat_part2), 0)
