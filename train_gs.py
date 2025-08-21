from __future__ import print_function, division
import sys
sys.path.append('mast3r/dust3r')
sys.path.append('mast3r')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = 'true'

import logging
import numpy as np
import cv2
import random
# import wandb
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator

from lib.human_loader import StereoHumanDataset, HumanDataset_GS
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render, pts2render_human
from lib.loss import l1_loss, ssim, psnr, render_depth_loss
from lib.utils import save_pts

import torch
import torch.optim as optim
import torch.distributed as dist
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, pred=True, dust3r=False)
        self.train_set = HumanDataset_GS(self.cfg.dataset, phase='train') # StereoHumanDataset HumanScene
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*4, pin_memory=True) # self.cfg.batch_size*2
        self.train_iterator = iter(self.train_loader)
        self.val_set = HumanDataset_GS(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100, final_div_factor=1e3, pct_start=0.01, cycle_momentum=False, anneal_strategy='cos') # 'linear'


        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0
        self.best_psnr = 0

        # self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False)

        self.model.train()


    def train(self):
        for idx in tqdm(range(self.total_steps, self.cfg.num_steps)):
            with accelerator.accumulate(self.model):
                
                self.optimizer.zero_grad()
                data = self.fetch_data(phase='train')

                data = self.model.forward_kdtree(data)

                Ll1_lrss, Lssim_lrss, Lpips_lrss = 0, 0, 0
                
                if idx < 2000:
                    mode = 'ss'
                    cur_view = 'novel_view' + str(random.randint(4, 5))
                elif idx < 4000:
                    mode = 'lr' 
                    cur_view = 'novel_view' + str(random.randint(1, 5))
                else:
                    mode = 'lrss'
                    cur_view = 'novel_view' + str(random.randint(1, 5))
                    
                # mode = 'lrss'
                # cur_view = 'novel_view' + str(random.randint(1, 5))
                
                if idx < 10000:
                    view_list = [cur_view]
                else:
                    view_list = ['novel_view1', 'novel_view2', 'novel_view3']
                
                for novel_name in view_list:
                    gt_novel = data[novel_name]['img'].cuda()
                    data = pts2render_human(data, bg_color=self.cfg.dataset.bg_color, mode=mode, name=novel_name)
                    render_novel = data[novel_name]['img_pred']

                    Ll1_lrss += l1_loss(render_novel, gt_novel)
                    Lssim_lrss += 1.0 - ssim(render_novel, gt_novel)

                loss = 0.8 * Ll1_lrss + 0.2 * Lssim_lrss # + 0.01 * Lpips_lrss
                loss = loss / len(view_list)
                    
                if idx < 2000:
                    valid_i = data['pred'][0]['mask']
                    loss_scale = data['pred'][0]['scale'][valid_i].mean()
                    loss += loss_scale * 2
                    
                if idx % 1000 == 0:
                    rgbs = torch.cat([render_novel, gt_novel], dim=-1)
                    cv2.imwrite(f'%s/%s.jpg' % (cfg.record.show_path_train, idx), rgbs[0].permute(1,2,0).cpu().detach().numpy()[...,::-1]*255)

                metrics = {
                    'l1_lrss': Ll1_lrss.item(),
                    'ssim_lrss': Lssim_lrss.item(),
                }

                if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0 and accelerator.is_main_process:
                    self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

                self.logger.push(metrics)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0 and accelerator.is_main_process: # and accelerator.is_main_process
                    self.model.eval()
                    cur_psnr = self.run_eval()
                    self.model.train()

                    if cur_psnr > self.best_psnr:
                        self.best_psnr = cur_psnr
                        self.save_ckpt(save_path=Path('%s/%s_best.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=True)

                self.total_steps += 1

        logging.info("FINISHED TRAINING")
        # self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))


    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list, psnr_list = [], [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():

                data = self.model.forward_kdtree(data)
                data = pts2render_human(data, bg_color=self.cfg.dataset.bg_color, mode='lrss', name='novel_view1')

                render_novel = data['novel_view1']['img_pred']
                gt_novel = data['novel_view1']['img'].cuda()
                # import pdb; pdb.set_trace()

                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if idx == show_idx:
                    tmp_novel = data['novel_view1']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path_val, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")

        torch.cuda.empty_cache()
        return val_psnr


    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                if item in ['instance', 'idx', 'mask']:
                    continue
                data[view][item] = data[view][item].cuda()
                
        return data


    def load_ckpt(self, load_path, load_optimizer=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)

        self.model.load_state_dict(ckpt['network'], strict=False)

        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1  
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")


    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    cfg = config()
    cfg.load("config/stage_gs.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2), str(dt.hour).zfill(2), str(dt.minute).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path_train = "experiments/%s/show/train" % cfg.exp_name
    cfg.record.show_path_val = "experiments/%s/show/val" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.record.log_path = "experiments/%s/log.txt" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path_train, cfg.record.show_path_val, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        filename=cfg.record.log_path
    )

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(3407)
    np.random.seed(3407)

    trainer = Trainer(cfg)
    if True:
        trainer.train()
    else:
        trainer.run_eval()

# CUDA_VISIBLE_DEVICES=0 python train_gs.py