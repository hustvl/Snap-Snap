from __future__ import print_function, division
import sys
sys.path.append('mast3r/dust3r')
sys.path.append('mast3r')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = 'true'

import argparse
import logging
import numpy as np
import cv2
import os
import time
from pathlib import Path
from tqdm import tqdm

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib, get_eval_calib, depth2pc
from lib.GaussianRender import pts2render, pts2render_human

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from lib.utils import *
def save_human(data, name):
    bs = data['lmain']['img'].shape[0]

    for i in range(bs):
        valid_i = data['pred'][i]['mask']
        xyz_i = pts_cam2world(data['pred'][i]['xyz'][valid_i].unsqueeze(0), data['lmain']['extr'][i].unsqueeze(0)).squeeze(0)
        rgb_i = data['pred'][i]['rgb'][valid_i] * 0.5 + 0.5
        rot_i = data['pred'][i]['rot'][valid_i]
        scale_i = data['pred'][i]['scale'][valid_i]
        opacity_i = data['pred'][i]['opacity'][valid_i]

        save_gaussians_as_ply(f'save_ply/{name}.ply', xyz_i, rgb_i, rot_i, scale_i, opacity_i)



class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, dust3r=True, pred=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

        self.time_list = []

    def infer_static(self, view_select, novel_view_nums):
        total_samples = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        for idx in tqdm(range(total_samples)):

            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)
            data['lmain']['img'] = data['lmain']['img'].unsqueeze(0)
            data['rmain']['img'] = data['rmain']['img'].unsqueeze(0)
            data['lmain']['extr'] = data['lmain']['extr'].unsqueeze(0)

            with torch.no_grad():

                start_time = time.time()
                
                data = self.model.forward_demo(data)
                # data = self.model.forward_demo_wo_side(data)
                
                self.time_list.append(time.time() - start_time)

                for eval_idx in range(novel_view_nums):
                    
                    data_i = get_eval_calib(data, self.cfg.dataset, data['name'], eval_idx)
                    data_i = pts2render_human(data_i, bg_color=self.cfg.dataset.bg_color, mode='lrss')

                    render_novel = self.tensor2np(data_i['novel_view']['img_pred'])
                    cv2.imwrite(self.cfg.test_out_path + '/%s_novel%s.jpg' % (data_i['name'], str(eval_idx).zfill(2)), render_novel)
                    
                    
    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

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
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=False)
        logging.info(f"Parameter loading done")
        
        self.model.load_dust3r_ckpt(self.cfg.dust3r_ckpt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, default='/data/lujia/val_data/render_data_2.0_1024_30_val')
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage_eval.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.dataset.use_hr_img = True
    cfg.test_out_path = 'vis/ours'
    cfg.dataset.bg_color = [1,1,1]
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    input_views = ['0', '15']
    novel_view_nums = 30

    render = StereoHumanRender(cfg, phase='test')
    render.infer_static(view_select=input_views, novel_view_nums=novel_view_nums)

    print('Average time: ', np.mean(render.time_list[20:]))
