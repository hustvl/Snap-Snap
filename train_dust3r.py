from __future__ import print_function, division
import sys
sys.path.append('mast3r/dust3r')
sys.path.append('mast3r')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = 'true'

import logging
import numpy as np
import cv2
# import wandb
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from lib.human_loader import StereoHumanDataset
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

# import lpips
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

# from torch.profiler import profile, record_function, ProfilerActivity

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# dist.init_process_group("nccl", rank=0, world_size=2)

accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, dust3r=True)
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=False,
                                       num_workers=self.cfg.batch_size*4, pin_memory=True) # self.cfg.batch_size*2
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)

        # dust3r_params = list(map(id, self.model.dust3r.parameters()))
        # base_params = filter(lambda p: id(p) not in dust3r_params, self.model.parameters())
        # self.optimizer = optim.AdamW([
        #     {'params': base_params},
        #     {'params': self.model.dust3r.parameters(), 'lr': self.cfg.lr * 0.1}
        # ], lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100, final_div_factor=1e4, pct_start=0.01, cycle_momentum=False, anneal_strategy='cos') # 'linear'

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0
        self.add_inpainting_loss = False
        self.best_psnr = 10

        # self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False)

        self.model.train()


    def train(self):
        for idx in tqdm(range(self.total_steps, self.cfg.num_steps)):
            with accelerator.accumulate(self.model):

                self.optimizer.zero_grad()
                data = self.fetch_data(phase='train')

                data, loss = self.model.forward_dust3r_wo_side(data)
                metrics = {
                    'reg': loss['reg'].item(),
                    'conf': loss['conf'].item(),
                }
                loss = loss['reg'] + loss['conf'] * 0.5

                # -------------------------------------------------

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

                    if cur_psnr < self.best_psnr:
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

        for idx in range(50):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                data, loss = self.model.forward_dust3r_wo_side(data)
                loss = loss['reg'] + loss['conf'] * 0.5
                psnr_list.append(loss.item())

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): dust3r-loss {val_psnr}")

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

        # data['pts3d'] = data['pts3d'].cuda()
        for view in ['lmain', 'rmain']:

            for item in data[view].keys():
                if item in ['instance', 'idx']:
                    continue
                data[view][item] = data[view][item].cuda()
                
        return data


    def load_ckpt(self, load_path, load_optimizer=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)

        self.model.load_state_dict(ckpt['network'], strict=False)
        # del ckpt

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
    cfg.load("config/stage_dust3r.yaml")
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

# CUDA_VISIBLE_DEVICES=0 python train_stage2.py