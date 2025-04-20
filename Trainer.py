import sys
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model.loss import *
from model.warplayer import warp
from config import *


class Model:
    def __init__(self, local_rank, context_aware_granularity):
        if context_aware_granularity is not None:
            MODEL_CONFIG = {
                'LOGNAME': 'GaraMoSt',
                'MODEL_TYPE': (feature_extractor, flow_estimation),
                'MODEL_ARCH': init_model_config(
                    F = 32,
                    lambda_range='local',
                    depth = [2, 2, 2, 2, 4, 4],
                    lambda_r=context_aware_granularity
                )
            }
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss()
        self.ploss = Perceptual_Loss()
        self.styloss = Style_Loss()
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers = False)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, name = None, folder_path = None, full_path = None, rank = 0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if full_path is not None:
                self.net.load_state_dict(convert(torch.load(full_path, map_location=torch.device('cpu'))))
            else:
                if name is None:
                    name = self.name
                if folder_path is None:
                    self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl', map_location=torch.device('cpu'))))
                else:
                    self.net.load_state_dict(convert(torch.load(folder_path + f'/{name}.pkl', map_location=torch.device('cpu'))))

    def load_pretrain_weight(self, name = None, folder_path = None, full_path = None, rank = 0):
        if rank <= 0 :
            if full_path is not None:
                self.net.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))
            else:
                if name is None:
                    name = self.name
                if folder_path is None:
                    self.net.load_state_dict(torch.load(f'ckpt/{name}.pkl', map_location=torch.device('cpu')))
                else:
                    self.net.load_state_dict(torch.load(folder_path + f'/{name}.pkl', map_location=torch.device('cpu')))

    def save_model(self, name = None, folder_path = None, rank=0):
        if rank == 0:
            if name is None:
                name = self.name
            if folder_path is None:
                torch.save(self.net.state_dict(), f'ckpt/{name}.pkl')
            else:
                torch.save(self.net.state_dict(), folder_path + f'/{name}.pkl')

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            sf, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, sf, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, TTA = False, timestep = 0.5, fast_TTA = False):
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(input, timestep=timestep)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred = self.net(imgs, timestep=timestep)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2

    @torch.no_grad()
    def multi_inference(self, img0, img1, TTA = False, down_scale = 1.0, time_list=[], fast_TTA = False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            sf, mf = self.net.feature_bone(img0, img1)
            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                sfd, mfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6])

            pred_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, sf, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, sfd, mfd)
                    flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

                pred = self.net.coraseWarp_and_Refine(imgs, sf, flow, mask)
                pred_list.append(pred)

            return pred_list

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds_lst = infer(input)
            return [(preds_lst[i][0] + preds_lst[i][1].flip(1).flip(2))/2 for i in range(len(time_list))]

        preds = infer(imgs)
        if TTA is False:
            return [preds[i][0] for i in range(len(time_list))]
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0] + flip_pred[i][0].flip(1).flip(2))/2 for i in range(len(time_list))]

    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            flow, mask, merged, pred = self.net(imgs)
            loss_l1 = (self.lap(pred, gt)).mean()

            factor = 1.0 / len(merged)
            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * factor

            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else:
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0

    def multi_gts_update(self, imgs, gts, TimeStepList:list, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            loss_l1_all = 0
            preds = []

            for timestep, i in zip(TimeStepList, range(len(TimeStepList))):
                flow, mask, merged, pred = self.net(imgs, timestep)
                gt_index1 = i * 3
                gt_index2 = (i + 1) * 3
                loss_l1 = (self.lap(pred, gts[:, gt_index1 : gt_index2])).mean()

                loss_l1_all += loss_l1
                preds.append(pred)

                factor = 1.0 / len(merged)
                for merge in merged:
                    loss_l1_all += (self.lap(merge, gts[:, gt_index1 : gt_index2])).mean() * factor

            self.optimG.zero_grad()
            loss_l1_all.backward()
            self.optimG.step()
            return preds, loss_l1_all
        else:
            with torch.no_grad():
                preds = []

                for timestep in TimeStepList:
                    flow, mask, merged, pred = self.net(imgs, timestep)
                    preds.append(pred)

                return preds, 0

    def multi_gts_losses_update(self, imgs, gts, TimeStepList:list, vgg_model_file:str = '', losses_weight_schedules:list = [], now_epoch:int = 0, now_step:int = 0, learning_rate=0, training=True):
        '''
        vgg_model_file: VGG 19网络权重的MATLAB格式的文件路径
        losses_weight_schedules: 各个loss的衰减计划

        举例：
        losses_weight_schedules = [
            {'boundaries_epoch':[0], 'boundaries_step':[0], 'values':[1.0, 1.0]},
            {'boundaries_epoch':[10], 'boundaries_step':[24000], 'values':[1.0, 0.25]},
            {'boundaries_epoch':[10], 'boundaries_step':[24000], 'values':[0.0, 40.0]}]

        优先按boundaries_epoch指定的边界来划分不同loss的阶段性weight，若没有boundaries_epoch则按boundaries_step划分。
        在boundaries_epoch指定的epoch前，某loss的权重按values[0]算；之后按values[1]算。boundaries_step同理。
        '''
        def decide_values(losses_weight_schedules:list, now_epoch:int, now_step:int):
            '''
            根据当前的epoch、step(iter)数，以及设定的阶段权重计划，决定各个loss的权重
            '''
            l1_epoch = losses_weight_schedules[0]['boundaries_epoch']
            p_epoch = losses_weight_schedules[1]['boundaries_epoch']
            sty_epoch = losses_weight_schedules[2]['boundaries_epoch']

            l1_step = losses_weight_schedules[0]['boundaries_step']
            p_step = losses_weight_schedules[1]['boundaries_step']
            sty_step = losses_weight_schedules[2]['boundaries_step']

            if ((len(l1_epoch) != 0) & (len(p_epoch) != 0) & (len(sty_epoch) != 0)): # 优先按boundaries_epoch指定的边界来划分
                if now_epoch < l1_epoch[0]:
                    l1_value = losses_weight_schedules[0]['values'][0]
                else:
                    l1_value = losses_weight_schedules[0]['values'][1]
                if now_epoch < p_epoch[0]:
                    p_value = losses_weight_schedules[1]['values'][0]
                else:
                    p_value = losses_weight_schedules[1]['values'][1]
                if now_epoch < sty_epoch[0]:
                    sty_value = losses_weight_schedules[2]['values'][0]
                else:
                    sty_value = losses_weight_schedules[2]['values'][1]
            elif ((len(l1_step) != 0) & (len(p_step) != 0) & (len(sty_step) != 0)): # 否则按boundaries_step划分
                if  now_step < l1_step[0]:
                    l1_value = losses_weight_schedules[0]['values'][0]
                else:
                    l1_value = losses_weight_schedules[0]['values'][1]
                if now_step < p_step[0]:
                    p_value = losses_weight_schedules[1]['values'][0]
                else:
                    p_value = losses_weight_schedules[1]['values'][1]
                if now_step < sty_step[0]:
                    sty_value = losses_weight_schedules[2]['values'][0]
                else:
                    sty_value = losses_weight_schedules[2]['values'][1]
            else:
                print("输入的losses_weight_schedules不符合规范, 需满足:(1)各loss的boundaries_epoch不为空  或  (2)各loss的boundaries_step不为空!")
                sys.exit()

            return l1_value, p_value, sty_value

        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            loss_all = 0
            preds = []
            l1_weight, p_weight, sty_weight = decide_values(losses_weight_schedules, now_epoch, now_step)

            for timestep, i in zip(TimeStepList, range(len(TimeStepList))):
                flow, mask, merged, pred = self.net(imgs, timestep)
                gt_index1 = i * 3
                gt_index2 = (i + 1) * 3
                if l1_weight != 0:
                    loss_l1 = (self.lap(pred, gts[:, gt_index1 : gt_index2])).mean()
                    print("loss_l1", loss_l1)
                    loss_all += loss_l1 * l1_weight
                if p_weight != 0:
                    loss_p = (self.ploss(pred, gts[:, gt_index1 : gt_index2], vgg_model_file)).mean()
                    print("loss_p", loss_p)
                    loss_all += loss_p * p_weight
                if sty_weight != 0:
                    loss_style = (self.styloss(pred, gts[:, gt_index1 : gt_index2], vgg_model_file)).mean()
                    print("loss_style", loss_style)
                    loss_all += loss_style * sty_weight

                preds.append(pred)

                factor = 1.0 / len(merged)
                for merge in merged:
                    if l1_weight != 0:
                        loss_all += (self.lap(merge, gts[:, gt_index1 : gt_index2])).mean() * factor * l1_weight
                    if p_weight != 0:
                        loss_all += (self.ploss(merge, gts[:, gt_index1 : gt_index2], vgg_model_file)).mean() * factor * p_weight
                    if sty_weight != 0:
                        loss_all += (self.styloss(merge, gts[:, gt_index1 : gt_index2], vgg_model_file)).mean() * factor * sty_weight

            self.optimG.zero_grad()
            loss_all.backward()
            self.optimG.step()
            return preds, loss_all
        else:
            with torch.no_grad():
                preds = []
                if len(TimeStepList) == 1:
                    flow, mask, merged, pred = self.net(imgs, TimeStepList[0])
                    preds.append(pred)
                else:
                    img0, img1 = imgs[:, :3], imgs[:, 3:6]
                    try:
                        sf, mf = self.net.feature_bone(img0, img1)
                    except:
                        sf, mf = self.net.module.feature_bone(img0, img1)
                    for timestep in TimeStepList:
                        try:
                            flow, mask = self.net.calculate_flow(imgs, timestep, sf, mf)
                            pred = self.net.coraseWarp_and_Refine(imgs, sf, flow, mask)
                        except:
                            flow, mask = self.net.module.calculate_flow(imgs, timestep, sf, mf)
                            pred = self.net.module.coraseWarp_and_Refine(imgs, sf, flow, mask)
                        preds.append(pred)

                return preds, 0
