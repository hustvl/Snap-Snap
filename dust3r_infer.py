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
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train') # StereoHumanDataset HumanScene
        self.train_loader = DataLoader(self.train_set, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=True) # self.cfg.batch_size*2
        self.train_iterator = iter(self.train_loader)
        
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False)

        self.model.train()

    def run_eval(self):
        self.len_train = int(len(self.train_loader) / self.train_set.train_boost)  # real length of train set
        for idx in range(self.len_train):
            data = self.fetch_data(phase='train')
            with torch.no_grad():
                self.model.infer_dust3r(data)
        logging.info(f"Train Set Done!")

        # self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        # for idx in range(self.len_val):
        #     data = self.fetch_data(phase='val')
        #     with torch.no_grad():
        #         self.model.infer_dust3r(data)
        # logging.info(f"Val Set Done!")

        torch.cuda.empty_cache()


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



if __name__ == '__main__':
    cfg = config()
    cfg = cfg.get_cfg()
    
    cfg.dust3r_ckpt = 'experiments/dust3r_21_ch_0721_1059/ckpt/dust3r_21_ch_final.pth'
    cfg.dataset.source_id = [0, 12]
    cfg.dataset.side_id = [6, 18]
    cfg.dataset.data_root = 'render_data_ch_1024_24'

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    )

    torch.manual_seed(3407)
    np.random.seed(3407)

    trainer = Trainer(cfg)
    trainer.run_eval()

# CUDA_VISIBLE_DEVICES=1 python dust3r_infer.py