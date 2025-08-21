from __future__ import print_function, division
import sys
sys.path.append('mast3r/dust3r')
sys.path.append('mast3r')

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib.human_loader import HumanSample
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.GaussianRender import pts2render, gather_pts
from lib.utils import save_pts

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, dust3r=True, inpainting=True, pred=True)
        self.dataset = HumanSample(self.cfg.dataset)
        self.model.cuda()

        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

    def infer_seqence(self):

        view_data = self.fetch_data(self.dataset.get_item())
        cv2.imwrite('experiments/test_out/lmain.png', view_data['lmain']['img'][0].cpu().numpy().transpose(1, 2, 0)*255)
                
        with torch.no_grad():
            data = self.model.forward_demo(view_data)
            gather_pts(data)

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                if item in ['instance', 'idx']:
                    continue
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)
        self.model.load_state_dict(ckpt['network'], strict=False)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, default='real_data/0019')
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage_demo.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.data_root = arg.test_data_root
    cfg.test_out_path = 'experiments/test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    render.infer_seqence()

# CUDA_VISIBLE_DEVICES=1 python demo.py