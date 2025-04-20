import os
import cv2
import math
import time
import sys
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import ast
import datetime

from Trainer import Model
from dataset import HustDataset_multi_gts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *
from utils.pytorch_msssim import ssim_matlab

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]


def NowTime():
    datetime_now = datetime.datetime.now()
    datetime_timestamp = datetime.datetime.timestamp(datetime_now)

    datetime_obj = datetime.datetime.fromtimestamp(datetime_timestamp)
    time = datetime_obj.strftime('%Y{y}%m{m}%d{d}%H{h}%M{mm}%S{s}').format(y='-',m='-', d='_', h=':', mm=':', s='')
    return time


def get_learning_rate(step, gts_num, strategy = 'ori', base_lr = 2e-4, milestones = [500,2500], gamma = 0.1, continue_lr = 0.):
    if strategy == 'ori':
        if step < 2000:
            mul = step / 2000
            return 2e-4 * mul
        else:
            mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
            return (2e-4 - 2e-5) * mul + 2e-5
    elif strategy == 'ori-custom':
        if step < milestones[0]:
            mul = step / milestones[0]
            return base_lr * mul
        else:
            mul = np.cos((step - milestones[0]) / (300 * args.step_per_epoch - milestones[0]) * math.pi) * 0.5 + 0.5
            return (base_lr - base_lr * 0.1) * mul + base_lr * 0.1
    elif strategy == 'ori-for-continue':
        if step < milestones[0]:
            mul = step / milestones[0]
            return continue_lr + (base_lr - continue_lr) * mul
        else:
            mul = np.cos((step - milestones[0]) / (300 * args.step_per_epoch - milestones[0]) * math.pi) * 0.5 + 0.5
            return (base_lr - base_lr * 0.1) * mul + base_lr * 0.1
    elif strategy == '3-stage-decay':
        if step < milestones[0]:
            return base_lr
        elif step < milestones[1]:
            return base_lr * gamma
        else:
            return base_lr * gamma * gamma
    elif strategy == '2-stage-decay':
        if step < milestones[0]:
            return base_lr
        else:
            return base_lr * gamma


def train(model, rank, batch_size, data_path, txt_path, log_name, model_save_name, model_save_folder_path, TimeStepList:list, lr_strategy_sets:dict,
          vgg_model_file:str, losses_weight_schedules:list, eval_per_epochs:int = 1, save_model_per_epochs:int = 1, max_epochs:int = 20, note_print:str = '',
          if_eval_inEpoch1:bool = False, eval_per_iters_inEpoch1:int = 100, extra_eval_save_iters:list = [], extra_save_iters:list = [], eval_mode:bool = False):
    if rank == 0:
        writer = SummaryWriter(log_name + '/train_log')

    step = 0
    nr_eval = 0
    best = 0
    best_ave_psnr = 0
    best_epoch = 0

    dataset = HustDataset_multi_gts('train', data_path, txt_path, gts_num = len(TimeStepList))
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()

    dataset_val = HustDataset_multi_gts('test', data_path, txt_path, gts_num = len(TimeStepList))
    sampler_val = DistributedSampler(dataset_val)
    val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True, num_workers=16, drop_last=True, sampler=sampler_val)

    Extra_eval_save_iters = [int(x * args.step_per_epoch) for x in extra_eval_save_iters]
    Extra_save_iters = [int(x * args.step_per_epoch) for x in extra_save_iters]

    if eval_mode:
        now_ave_psnr, now_ave_ssim = evaluate(model, TimeStepList, val_data, nr_eval, rank, log_name, note_print, step = step, is_extra_eval = True, txt_save_path = log_name)
        print("\nnow_ave_psnr, now_ave_ssim: {} {}\n".format(now_ave_psnr, now_ave_ssim))
        try:
            with open(os.path.join(log_name, "Eval_SSIM-" + str(np.round(now_ave_ssim, 4)) + ",PSNR-" + str(np.round(now_ave_psnr, 4)) + ".txt"), "w+") as txt:
                txt.write("Average SSIM: " + str(now_ave_ssim) + "\n")
                txt.write("Average PSNR: " + str(now_ave_psnr) + "\n")
        except:
            print("*** txt file save erorr! ***")
        sys.exit()

    print(NowTime(),'Start training...')
    time_stamp = time.time()

    dist.barrier()

    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)

        for i, imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gts = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step, gts_num = len(TimeStepList), strategy = lr_strategy_sets['strategy'], base_lr = lr_strategy_sets['base_lr'], milestones = lr_strategy_sets['milestones'], gamma = lr_strategy_sets['gamma'],
                                              continue_lr = lr_strategy_sets['continue_lr'])
            _, loss = model.multi_gts_losses_update(imgs, gts, TimeStepList, vgg_model_file, losses_weight_schedules, epoch, step, learning_rate = learning_rate, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 10 == 1 and rank == 0:
                writer.add_scalar('learning_rate(step)', learning_rate, step)
                writer.add_scalar('loss(step)', loss, step)
            if rank == 0:
                print('{}  {}  Epoch:{}  {}/{}  Data-Time:{:.4f}  Train-Time:{:.4f}  Loss:{:.4e}  Now-Best-Epoch:{}'.format(NowTime(), note_print, epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss, best_epoch))

            if nr_eval < 1 and if_eval_inEpoch1 and (step % eval_per_iters_inEpoch1 == 1):
                dist.barrier()
                now_ave_psnr, now_ave_ssim = evaluate(model, TimeStepList, val_data, nr_eval, rank, log_name, note_print, if_eval_inEpoch1, step, txt_save_path = log_name)
                if now_ave_psnr > best_ave_psnr:
                    best_ave_psnr = now_ave_psnr
                    best_epoch = round(step / args.step_per_epoch, 2)
                    model.save_model(name = model_save_name + "--Best", folder_path = model_save_folder_path, rank = rank)
                dist.barrier()
            if step in Extra_eval_save_iters:
                dist.barrier()
                model.save_model(name = model_save_name + "--Iter" + str(step), folder_path = model_save_folder_path, rank = rank)
                now_ave_psnr, now_ave_ssim = evaluate(model, TimeStepList, val_data, nr_eval, rank, log_name, note_print, step = step, is_extra_eval = True, txt_save_path = log_name)
                if now_ave_psnr > best_ave_psnr:
                    best_ave_psnr = now_ave_psnr
                    best_epoch = round(step / args.step_per_epoch, 2)
                    model.save_model(name = model_save_name + "--Best", folder_path = model_save_folder_path, rank = rank)
                dist.barrier()
            if step in Extra_save_iters:
                try:
                    model.save_model(name = model_save_name + "--Iter" + str(step), folder_path = model_save_folder_path, rank = rank)
                except:
                    if rank == 0:
                        print("Iter: {} Model weight save failed.".format(str(step)))

            step += 1

        if rank == 0:
            writer.add_scalar('learning_rate(epoch)', learning_rate, epoch)
            writer.add_scalar('loss(epoch)', loss, epoch)

        nr_eval += 1

        if nr_eval % save_model_per_epochs == 0:
            model.save_model(name = model_save_name + "--epoch" + str(nr_eval), folder_path = model_save_folder_path, rank = rank)

        if nr_eval % eval_per_epochs == 0:
            now_ave_psnr, now_ave_ssim = evaluate(model, TimeStepList, val_data, nr_eval, rank, log_name, note_print, txt_save_path = log_name)
            if now_ave_psnr > best_ave_psnr:
                best_ave_psnr = now_ave_psnr
                best_epoch = nr_eval
                model.save_model(name = model_save_name + "--Best", folder_path = model_save_folder_path, rank = rank)

        if epoch == max_epochs:
            model.save_model(name = model_save_name + "--Last", folder_path = model_save_folder_path, rank = rank)

        dist.barrier()

    return best_epoch


def evaluate(model, TimeStepList, val_data, nr_eval, rank, log_name, note_print:str = '', if_eval_inEpoch1:bool = False, step:int = 120, is_extra_eval:bool = False, txt_save_path:str = None):
    total_time_start = time.time()
    if rank == 0:
        writer_val = SummaryWriter(log_name + '/eval_log')

    if (nr_eval < 1 and if_eval_inEpoch1) or is_extra_eval:
        txt_log = open(txt_save_path + "/Log_Eval-Iter" + str(step) + '_' + NowTime() + ".txt", 'w')
    else:
        txt_log = open(txt_save_path + "/Log_Eval-Epoch" + str(nr_eval) + '_' + NowTime() + ".txt", 'w')

    # Listing of initialization for each intermediate frame, adding more frames if needed
    psnr_1, psnr_2, psnr_3, psnr_4, psnr_5, psnr_6, psnr_7 = [], [], [], [], [], [], []
    psnr_all = [psnr_1, psnr_2, psnr_3, psnr_4, psnr_5, psnr_6, psnr_7]
    ssim_1, ssim_2, ssim_3, ssim_4, ssim_5, ssim_6, ssim_7 = [], [], [], [], [], [], []
    ssim_all = [ssim_1, ssim_2, ssim_3, ssim_4, ssim_5, ssim_6, ssim_7]

    time_list = []

    error_count = 0
    num = val_data.__len__()
    time_stamp = time.time()

    for index, imgs in enumerate(val_data):
        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()

        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gts = imgs[:, 0:6], imgs[:, 6:]

        with torch.no_grad():
            try:
                preds, _ = model.multi_gts_losses_update(imgs, gts, TimeStepList, training = False)
            except:
                print('***** Error Evaluating, Skipped! *****  Batch:{}  {}/{}'.format(index, index, num))
                preds = []
                if rank == 0:
                    error_count += 1
                continue
            eva_time_interval = time.time() - time_stamp
            time_list.append(eva_time_interval)
            time_stamp = time.time()

        for pred, i in zip(preds, range(len(preds))):
            gt_index1 = i * 3
            gt_index2 = (i + 1) * 3
            gt = gts[:, gt_index1 : gt_index2]

            for j in range(gt.shape[0]):
                psnr_all[i].append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
                ssim_all[i].append(ssim_matlab(gt[j].cuda().unsqueeze(0), pred[j].unsqueeze(0)).detach().cpu().numpy())

        if (nr_eval < 1 and if_eval_inEpoch1) or is_extra_eval:
            try:
                print('{}  {}  After Iter:{}  Evaluating... Batch:{}  {}/{}  Data-Time:{:.4f}  Eval-Time:{:.4f}  PSNR_of_Inter_Frames:{},  SSIM_of_Inter_Frames:{},  All-errors-so-far:{}'.format(
                    NowTime(), note_print, step, index, index, num, data_time_interval, eva_time_interval, [np.round(psnr_all[i][-1], 4) for i in range(len(TimeStepList))],
                    [np.round(ssim_all[i][-1], 4) for i in range(len(TimeStepList))], error_count))
                txt_log.writelines('{}  {}  After Iter:{}  Evaluating... Batch:{}  {}/{}  Data-Time:{:.4f}  Eval-Time:{:.4f}  PSNR_of_Inter_Frames:{},  SSIM_of_Inter_Frames:{},  All-errors-so-far:{}'.format(
                    NowTime(), note_print, step, index, index, num, data_time_interval, eva_time_interval, [np.round(psnr_all[i][-1], 4) for i in range(len(TimeStepList))],
                    [np.round(ssim_all[i][-1], 4) for i in range(len(TimeStepList))], error_count))
                txt_log.write('\n')
            except:
                print('*^*^*^*^* Error Results Reporting, Skipped! *^*^*^*^*')
                txt_log.writelines('*^*^*^*^* Error Results Reporting, Skipped! *^*^*^*^*')
                txt_log.write('\n')
        else:
            try:
                print('{}  {}  After Epoch:{}  Evaluating... Batch:{}  {}/{}  Data-Time:{:.4f}  Eval-Time:{:.4f}  PSNR_of_Inter_Frames:{},  SSIM_of_Inter_Frames:{},  All-errors-so-far:{}'.format(
                    NowTime(), note_print, nr_eval, index, index, num, data_time_interval, eva_time_interval, [np.round(psnr_all[i][-1], 4) for i in range(len(TimeStepList))],
                    [np.round(ssim_all[i][-1], 4) for i in range(len(TimeStepList))], error_count))
                txt_log.writelines('{}  {}  After Epoch:{}  Evaluating... Batch:{}  {}/{}  Data-Time:{:.4f}  Eval-Time:{:.4f}  PSNR_of_Inter_Frames:{},  SSIM_of_Inter_Frames:{},  All-errors-so-far:{}'.format(
                    NowTime(), note_print, nr_eval, index, index, num, data_time_interval, eva_time_interval, [np.round(psnr_all[i][-1], 4) for i in range(len(TimeStepList))],
                    [np.round(ssim_all[i][-1], 4) for i in range(len(TimeStepList))], error_count))
                txt_log.write('\n')
            except:
                print('*^*^*^*^* Error Results Reporting, Skipped! *^*^*^*^*')
                txt_log.writelines('*^*^*^*^* Error Results Reporting, Skipped! *^*^*^*^*')
                txt_log.write('\n')

    psnr_all_mean_list = [np.array(psnr_all[i]).mean() for i in range(len(TimeStepList))]
    psnr_all_mean_list_for_print = [np.round(np.array(psnr_all[i]).mean(), 4) for i in range(len(TimeStepList))]

    ssim_all_mean_list = [np.array(ssim_all[i]).mean() for i in range(len(TimeStepList))]
    ssim_all_mean_list_for_print = [np.round(np.array(ssim_all[i]).mean(), 4) for i in range(len(TimeStepList))]

    psnr_all_std_list = [np.array(psnr_all[i]).std() for i in range(len(TimeStepList))]
    psnr_all_std_list_for_print = [np.round(np.array(psnr_all[i]).std(), 4) for i in range(len(TimeStepList))]

    ssim_all_std_list = [np.array(ssim_all[i]).std() for i in range(len(TimeStepList))]
    ssim_all_std_list_for_print = [np.round(np.array(ssim_all[i]).std(), 4) for i in range(len(TimeStepList))]

    total_time_interval = time.time() - total_time_start

    if rank == 0:
        if (nr_eval < 1 and if_eval_inEpoch1) or is_extra_eval:
            print("--------------------------------------------------------------------------------------------------------------------------")
            print('{}  {}  After Iter:{}  Evaluate Over! Eval-Total-Time:{:.4f}  All-Batch-Mean-PSNR_of_Inter_Frames:{},  All-Batch-Mean-SSIM_of_Inter_Frames:{},  All-Batch-STD-PSNR_of_Inter_Frames:{},  All-Batch-STD-SSIM_of_Inter_Frames:{},  Mean-Eval-Time-for-{}-frame:{:.4f},  All-errors-so-far:{}'.format(NowTime(), note_print, step, total_time_interval, psnr_all_mean_list_for_print, ssim_all_mean_list_for_print, psnr_all_std_list_for_print, ssim_all_std_list_for_print, len(TimeStepList), np.array(time_list).mean(), error_count))
            print("--------------------------------------------------------------------------------------------------------------------------")
            txt_log.writelines('{}  {}  After Iter:{}  Evaluate Over! Eval-Total-Time:{:.4f}  All-Batch-Mean-PSNR_of_Inter_Frames:{},  All-Batch-Mean-SSIM_of_Inter_Frames:{},  All-Batch-STD-PSNR_of_Inter_Frames:{},  All-Batch-STD-SSIM_of_Inter_Frames:{},  Mean-Eval-Time-for-{}-frame:{:.4f},  All-errors-so-far:{}'.format(NowTime(), note_print, step, total_time_interval, psnr_all_mean_list_for_print, ssim_all_mean_list_for_print, psnr_all_std_list_for_print, ssim_all_std_list_for_print, len(TimeStepList), np.array(time_list).mean(), error_count))
            txt_log.write('\n')
            for i in range(len(psnr_all_mean_list)):
                writer_val.add_scalar('psnr_{}_mean(iter)'.format(i), psnr_all_mean_list[i], step)
                writer_val.add_scalar('ssim_{}_mean(iter)'.format(i), ssim_all_mean_list[i], step)
                writer_val.add_scalar('psnr_{}_std(iter)'.format(i), psnr_all_std_list[i], step)
                writer_val.add_scalar('ssim_{}_std(iter)'.format(i), ssim_all_std_list[i], step)
            writer_val.add_scalar('All-errors-so-far(iter)', error_count, step)
            txt_tag = 'Iter' + str(step) + '_'
        else:
            print("--------------------------------------------------------------------------------------------------------------------------")
            print('{}  {}  After Epoch:{}  Evaluate Over! Eval-Total-Time:{:.4f}  All-Batch-Mean-PSNR_of_Inter_Frames:{},  All-Batch-Mean-SSIM_of_Inter_Frames:{},  All-Batch-STD-PSNR_of_Inter_Frames:{},  All-Batch-STD-SSIM_of_Inter_Frames:{},  Mean-Eval-Time-for-{}-frame:{:.4f},  All-errors-so-far:{}'.format(NowTime(), note_print, nr_eval, total_time_interval, psnr_all_mean_list_for_print, ssim_all_mean_list_for_print, psnr_all_std_list_for_print, ssim_all_std_list_for_print, len(TimeStepList), np.array(time_list).mean(), error_count))
            print("--------------------------------------------------------------------------------------------------------------------------")
            txt_log.writelines('{}  {}  After Epoch:{}  Evaluate Over! Eval-Total-Time:{:.4f}  All-Batch-Mean-PSNR_of_Inter_Frames:{},  All-Batch-Mean-SSIM_of_Inter_Frames:{},  All-Batch-STD-PSNR_of_Inter_Frames:{},  All-Batch-STD-SSIM_of_Inter_Frames:{},  Mean-Eval-Time-for-{}-frame:{:.4f},  All-errors-so-far:{}'.format(NowTime(), note_print, nr_eval, total_time_interval, psnr_all_mean_list_for_print, ssim_all_mean_list_for_print, psnr_all_std_list_for_print, ssim_all_std_list_for_print, len(TimeStepList), np.array(time_list).mean(), error_count))
            txt_log.write('\n')
            for i in range(len(psnr_all_mean_list)):
                writer_val.add_scalar('psnr_{}_mean(epoch)'.format(i), psnr_all_mean_list[i], nr_eval)
                writer_val.add_scalar('ssim_{}_mean(epoch)'.format(i), ssim_all_mean_list[i], nr_eval)
                writer_val.add_scalar('psnr_{}_std(epoch)'.format(i), psnr_all_std_list[i], nr_eval)
                writer_val.add_scalar('ssim_{}_std(epoch)'.format(i), ssim_all_std_list[i], nr_eval)
            writer_val.add_scalar('All-errors-so-far(epoch)', error_count, nr_eval)
            txt_tag = 'Epoch' + str(nr_eval) + '_'

        for i in range(len(TimeStepList)):
            f1 = open(txt_save_path + "/" + txt_tag + "ssim_{}.txt".format(i), 'w+')
            for ssim in ssim_all[i]:
                f1.write(str(ssim) + '\n')
            f1.close()

            f2 = open(txt_save_path + "/" + txt_tag + "psnr_{}.txt".format(i), 'w+')
            for psnr in psnr_all[i]:
                f2.write(str(psnr) + '\n')
            f2.close()

        f1_all = open(txt_save_path + "/" + txt_tag + "ssim_all.txt", 'w+')
        for j in range(len(ssim_all[0])):
            for i in range(len(TimeStepList)):
                f1_all.write(str(ssim_all[i][j]) + "    ")
            f1_all.write('\n')
        f1_all.close()

        f2_all = open(txt_save_path + "/" + txt_tag + "psnr_all.txt", 'w+')
        for j in range(len(psnr_all[0])):
            for i in range(len(TimeStepList)):
                f2_all.write(str(psnr_all[i][j]) + "    ")
            f2_all.write('\n')
        f2_all.close()

    txt_log.close()

    return np.mean(psnr_all_mean_list), np.mean(ssim_all_mean_list)


def generate_sv_points(start:float, end:float, inter:float):
    plist = []
    now = 0.0
    while now < end:
        plist.append(now)
        now = round(now + inter, 3)
    return plist


if __name__ == "__main__":
    # ------------------------------------------------------
    # The cycle of saving and evaluating models
    eval_per_epochs = 1 # How many epoch to evaluate once
    save_model_per_epochs = 1 # How many epoch to save once

    # Used to train a large amount of data, observing the convergence within the first epoch to assist in finding the appropriate x (number of iters) in the loss function strategy.
    if_eval_inEpoch1 = False # Whether to add multiple evaluations within the first epoch; If yes, the period is specified by the 'eval_per_iters_inEpoch1'.
    eval_per_iters_inEpoch1 = 1000

    # Used for training large amounts of data to observe metrics and save weights at additional custom moments
    ## The model is evaluated and saved when the number of iter trained = extra_eval_save_iters (e.g. x) is multiplied by args.step_per_epoch (e.g. x * args.step_per_epoch)
    extra_eval_save_iters = [] # If there is no special need, the default is an empty list

    # Used to save weights during additional custom moments when training large amounts of data
    # extra_save_iters = generate_sv_points(0, 5, 0.022) # Start with 0 epochs and end with 5 epochs at an interval of 0.022 epochs
    extra_save_iters = [] # If there is no special need, the default is an empty list

    # Fill in the path where the imagenet-vgg-verydeep-19.mat is located
    vgg_weight_path ='./weights/vgg_weight/imagenet-vgg-verydeep-19.mat' # Used to calculate perceptual loss, style loss
    # -----------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of datasets', required=True)
    parser.add_argument('--txt_path', type=str, help='The root directory of the txt file that stores the training set list and test set list.', required=True)
    parser.add_argument('--inter_frames', type=int, required=True)
    parser.add_argument('--note', default=None, type=str, help='This is used to describe the experiment. The folder name will be used when saving.')
    parser.add_argument('--model_save_name', type=str, default='GaraMoSt', help='Name when saving the model file.')
    parser.add_argument('--lr_strategy_sets', type=ast.literal_eval, default={'strategy': 'ori', 'base_lr': 2e-4, 'milestones': [2000], 'gamma': 0.1, 'continue_lr': 0.}, help='Learning rate strategy.')
    parser.add_argument('--losses_weight_schedules', type=ast.literal_eval, default=[{'boundaries_epoch':[0], 'boundaries_step':[], 'values':[1.0, 1.0]},
                                                                                    {'boundaries_epoch':[0], 'boundaries_step':[], 'values':[1.0, 0.25]},
                                                                                    {'boundaries_epoch':[2], 'boundaries_step':[], 'values':[0.0, 40.0]}],
                        help='Multiple loss weighting strategies.') # if finetune, 'boundaries_epoch' 0-0-0 is recommended
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of Epoch trained.') # GaraMoSt can quickly achieve good results in the early stage of training, but complete convergence usually occurs around 90 epochs
    parser.add_argument('--note_print', default='', type=str, help='This is used to describe the experiment. It will be displayed in real time during training.')
    parser.add_argument('--pretrain_weight', default=None, type=str, help='Pre training weights. Full path containing pkl suffix.')
    parser.add_argument('--multi_nodes', default=False, type=bool, help='Whether to use multi-node (machine) training.')
    parser.add_argument('--eval_mode', default=False, type=bool, help='Whether to perform only the evaluation and end. If true, fill "pretrain_weight" to evaluate.')
    parser.add_argument('--context_aware_granularity', type=ast.literal_eval, default={'high-level': 7, 'low-level': 7}, help='Context-aware granularity at different scales.') # Adjust according to the different frame count of interpolation
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)

    global_rank = torch.distributed.get_rank()
    if args.multi_nodes: # Multi-machine multi-GPU
        rank = global_rank
    else: # Single-machine multi/single-GPU
        rank = args.local_rank
    torch.cuda.set_device(args.local_rank)

    Interframe_num = int(args.inter_frames)
    if Interframe_num == 1:
        TimeStepList = [0.5]
    elif Interframe_num == 2:
        TimeStepList = [0.3333333333333333, 0.6666666666666667]
    elif Interframe_num == 3:
        TimeStepList = [0.25, 0.50, 0.75]
    else:
        print("'inter_frames' invalid. Currently, 1, 2, and 3 frames are supported. You can also try training a model that interpolates more frames.")
        sys.exit()

    if args.note is None:
        log_name = args.txt_path + '/log_' + NowTime()
    else:
        log_name = args.txt_path + '/log_' + str(args.note) + "_" + NowTime()

    model_save_folder_path = os.path.join(log_name, 'Saved_Model')
    if rank == 0:
        if not os.path.exists(log_name):
            os.mkdir(log_name)
        if not os.path.exists(model_save_folder_path):
            os.mkdir(model_save_folder_path)

    seed = 9182
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    model = Model(args.local_rank, [int(args.context_aware_granularity['high-level']), int(args.context_aware_granularity['low-level'])])
    if args.pretrain_weight is not None:
        model.load_pretrain_weight(full_path = args.pretrain_weight)
        if rank == 0:
            print("---------------\n The pre training weights have been loaded, and the path is as follow:\n      {}\n---------------".format(args.pretrain_weight))

    best_epoch = train(model = model, rank = rank, batch_size = args.batch_size, data_path = args.data_path, txt_path = args.txt_path,
            log_name = log_name, model_save_name = args.model_save_name, model_save_folder_path = model_save_folder_path, TimeStepList = TimeStepList,
            lr_strategy_sets = args.lr_strategy_sets, vgg_model_file = vgg_weight_path, losses_weight_schedules = args.losses_weight_schedules,
            eval_per_epochs = eval_per_epochs, save_model_per_epochs = save_model_per_epochs, max_epochs = args.max_epochs, note_print = args.note_print,
            if_eval_inEpoch1 = if_eval_inEpoch1, eval_per_iters_inEpoch1 = eval_per_iters_inEpoch1, extra_eval_save_iters = extra_eval_save_iters,
            extra_save_iters = extra_save_iters, eval_mode = args.eval_mode)

    if rank == 0:
        print("The epoch corresponding to the best model file is: ", best_epoch)
