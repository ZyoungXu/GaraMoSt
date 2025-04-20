import sys
import os
import fnmatch
import numpy as np
import cv2
import math
import torch
import argparse

import config as cfg
from Trainer import Model
from utils.padder import InputPadder


def run_interpolator(model, Frame1, Frame2, time_list, Output_Frames_list, TTA = True):
    I0 = cv2.imread(Frame1)
    I2 = cv2.imread(Frame2)

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    preds = model.multi_inference(I0_, I2_, TTA = TTA, time_list = time_list, fast_TTA = TTA)

    for pred, Output_Frame in zip(preds, Output_Frames_list):
        mid_image = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(Output_Frame, mid_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./weights/checkpoints/inf1_r7-r7_best.pkl')
    parser.add_argument('--frame1', type=str, default='./demo_images/DSA_a.png')
    parser.add_argument('--frame2', type=str, default='./demo_images/DSA_b.png')
    parser.add_argument('--inter_frames', type=int, default=1)
    args = parser.parse_args()

    model_path = args.model_path
    inf_folder_path = os.path.dirname(args.frame1)
    Interframe_num = args.inter_frames

    if Interframe_num == 1:
        TimeStepList = [0.5]
        Inter_Frames_list = [inf_folder_path + "//" + 'InferImage.png']
        TTA = True
        context_aware_granularity=[7, 7]
    elif Interframe_num == 2:
        TimeStepList = [0.3333333333333333, 0.6666666666666667]
        Inter_Frames_list = [inf_folder_path + "//" + 'InferImage_1_in_2.png',
                            inf_folder_path + "//" + 'InferImage_2_in_2.png']
        TTA = True
        context_aware_granularity=[7, 29]
    elif Interframe_num == 3:
        TimeStepList = [0.25, 0.50, 0.75]
        Inter_Frames_list = [inf_folder_path + "//" + 'InferImage_1_in_3.png',
                            inf_folder_path + "//" + 'InferImage_2_in_3.png',
                            inf_folder_path + "//" + 'InferImage_3_in_3.png']
        TTA = True
        context_aware_granularity=[7, 29]
    else:
        print("'inter_frames' invalid. Currently, 1, 2, and 3 frames are supported. You can also try training a model that interpolates more frames.")
        sys.exit()

    model = Model(-1, context_aware_granularity)
    model.load_model(full_path = model_path)
    model.eval()
    model.device()

    run_interpolator(model, args.frame1, args.frame2, time_list = TimeStepList, Output_Frames_list = Inter_Frames_list, TTA = TTA)
