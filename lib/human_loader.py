from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image
import cv2
import math
import torch
import torch.nn as nn
from plyfile import PlyData
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import flow2depth, depth2pc, save_pts
from pathlib import Path
import logging
import json
from tqdm import tqdm

# import sys
# sys.path.append('submodules/SemanticGuidedHumanMatting')
# from model.model import HumanSegment, HumanMatting
# import inference

# from smplx.body_models import SMPLX


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm


def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                          torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)  # 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]

    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    offset0 = new_intr1[0, 2] - new_intr0[0, 2]
    disparity0 = -new_depth0 * Tf_x
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1


def read_img(name):
    img = np.array(Image.open(name))
    return img

def read_img_demo(name):
    img = np.array(cv2.imread(name))[..., ::-1]
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


class StereoHumanDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        if self.phase in ['train', 'val']:
            self.data_root = os.path.join(opt.data_root, self.phase)
        elif self.phase == 'test':
            self.data_root = opt.test_data_root

        self.img_path = os.path.join(self.data_root, 'img/%s/%s.jpg') #'%d'
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%s_hr.jpg') #'%d'
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%s.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%s.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%s_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%s_extrinsic.npy')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))

        self.sample_list = list(os.listdir(os.path.join(self.data_root, 'img')))
        if self.phase == 'train' and self.opt.add_2_1:
            self.sample_list += list(os.listdir(os.path.join(self.data_root.replace('2.0', '2.1'), 'img')))
        if self.phase == 'train' and self.opt.add_ch:
            self.sample_list += list(os.listdir(os.path.join(self.data_root.replace('2.0', 'ch'), 'img')))
        self.sample_list = sorted(self.sample_list)
        
        self.num_view = None
        if self.opt.source_id == 'all':
            self.num_view = int(opt.data_root.split('_')[-1])


    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True, require_pts=True, half_res=False):
        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        depth_name = self.depth_path % (sample_name, source_id)
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)

        if 2500 > int(sample_name.split('_')[0]) >= 526 and self.opt.add_2_1:
            img_name = img_name.replace('2.0', '2.1')
            image_hr_name = image_hr_name.replace('2.0', '2.1')
            mask_name = mask_name.replace('2.0', '2.1')
            depth_name = depth_name.replace('2.0', '2.1')
            intr_name = intr_name.replace('2.0', '2.1')
            extr_name = extr_name.replace('2.0', '2.1')
            
        if int(sample_name.split('_')[0]) >= 2500 and self.opt.add_ch:
            img_name = img_name.replace('2.0', 'ch')
            image_hr_name = image_hr_name.replace('2.0', 'ch')
            mask_name = mask_name.replace('2.0', 'ch')
            depth_name = depth_name.replace('2.0', 'ch')
            intr_name = intr_name.replace('2.0', 'ch')
            extr_name = extr_name.replace('2.0', 'ch')

        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts, depth, smpl_pts = None, None, None, None

        if require_mask:
            mask = read_img(mask_name)

        depth = read_depth(depth_name)
        if require_pts and os.path.exists(depth_name):
            pts = depth2pts(torch.FloatTensor(depth), torch.FloatTensor(extr), torch.FloatTensor(intr))
        
        if hr_img:
            img = read_img(image_hr_name)
            intr[:2] *= 2
        else:
            img = read_img(img_name)

        if half_res:
            img = img[::2, ::2]
            intr[:2] /= 2

            if require_mask and not hr_img:
                mask = mask[::2, ::2]
            
            if require_pts:
                if not hr_img:
                    pts = pts[::2, ::2]

        pos_map = None
        return img, mask, intr, extr, pts, depth, pos_map


    def get_novel_view_tensor(self, sample_name, view_id, half_res=False):
        # import pdb;pdb.set_trace()
        img, mask, intr, extr, pts, depth, pos_map = self.load_single_view(sample_name, view_id, hr_img=self.opt.use_hr_img,
                                                                require_mask=True, require_pts=True, half_res=half_res)
        width, height = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        mask = mask / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': torch.FloatTensor(mask),
            'depth': torch.FloatTensor(depth),
            'pts3d': torch.FloatTensor(pts),
            'intr': torch.FloatTensor(intr),
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        return novel_view_data


    def get_rectified_stereo_data_none(self, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0, pts0, depth0, smpl_pts0 = main_view_data
        img1, mask1, intr1, extr1, pts1, depth1, smpl_pts1 = ref_view_data

        camera = {
            'intr0': intr0,
            'intr1': intr1,
            'extr0': extr0,
            'extr1': extr1,
            'Tf_x': None
        }

        stereo_data = {
            'img0': img0,
            'mask0': mask0,
            'depth0': depth0,
            'pts0': pts0,
            'pos_map0': smpl_pts0,
            'img1': img1,
            'mask1': mask1,
            'depth1': depth1,
            'camera': camera,
            'pts1': pts1,
            'pos_map1': smpl_pts1
        }

        return stereo_data


    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'depth': torch.FloatTensor(stereo_data['depth0']),
            'pts3d': torch.FloatTensor(stereo_data['pts0']),
            'idx': 0,
            'instance': 'left',
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'depth': torch.FloatTensor(stereo_data['depth1']),
            'pts3d': torch.FloatTensor(stereo_data['pts1']),
            'idx': 1,
            'instance': 'right',
        }

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}


    def get_item(self, index, side_id=[4, 9], novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        half_res = False
        
        if self.num_view:
            view0_idx = np.random.choice(list(range(self.num_view)))
            view1_idx = (view0_idx + self.num_view // 2) % self.num_view
            side_id = [(view0_idx + self.num_view // 4) % self.num_view, (view1_idx + self.num_view // 4) % self.num_view]
            
        else:
            view0_idx, view1_idx = self.opt.source_id[0], self.opt.source_id[1]

        view0_data = self.load_single_view(sample_name, view0_idx, hr_img=False,
                                            require_mask=True, require_pts=True, half_res=half_res)
        view1_data = self.load_single_view(sample_name, view1_idx, hr_img=False,
                                            require_mask=True, require_pts=True, half_res=half_res)
        stereo_np = self.get_rectified_stereo_data_none(main_view_data=view0_data, ref_view_data=view1_data)

        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        dict_tensor.update({
            'side_view1': self.get_novel_view_tensor(sample_name, side_id[0], half_res=half_res),
            'side_view2': self.get_novel_view_tensor(sample_name, side_id[1], half_res=half_res),
        })

        if novel_id:
            novel_id = np.random.choice(novel_id)
            dict_tensor.update({
                'novel_view': self.get_novel_view_tensor(sample_name, novel_id, half_res=half_res),
            })

        return dict_tensor


    def get_test_item(self, index, source_id, half_res=False):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        view0_data = self.load_single_view(sample_name, source_id[0], hr_img=False, require_mask=True, require_pts=True, half_res=False)
        view1_data = self.load_single_view(sample_name, source_id[1], hr_img=False, require_mask=True, require_pts=True, half_res=False)
        stereo_np = self.get_rectified_stereo_data_none(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        img_len = 2048 if self.opt.use_hr_img else 1024
        # img_len = img_len // 2 if half_res else img_len
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })

        return dict_tensor

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, side_id=self.opt.side_id, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, side_id=self.opt.side_id, novel_id=self.opt.val_novel_id)

    # def __len__(self):
    #     return len(self.sample_list)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)



class HumanDataset_GS(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        if self.phase in ['train', 'val']:
            self.data_root = os.path.join(opt.data_root, self.phase)
        elif self.phase == 'test':
            self.data_root = opt.test_data_root

        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%d_hr.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%d_extrinsic.npy')

        self.sample_list = list(os.listdir(os.path.join(self.data_root, 'img')))
        if self.phase == 'train' and self.opt.add_2_1:
            self.sample_list += list(os.listdir(os.path.join(self.data_root.replace('2.0', '2.1'), 'img')))
        if self.phase == 'train' and self.opt.add_ch:
            self.sample_list += list(os.listdir(os.path.join(self.data_root.replace('2.0', 'ch'), 'img')))
        self.sample_list = sorted(self.sample_list)
        print('Sample list length:', len(self.sample_list))

        if self.opt.train_novel_id == 'all':
            num_view = int(opt.data_root.replace('_noise', '').split('_')[-1])
            self.train_novel_id = list(range(0, num_view, 1))
        else:
            self.train_novel_id = self.opt.train_novel_id

        if self.opt.val_novel_id == 'all':
            num_view = int(opt.data_root.split('_')[-1])
            self.val_novel_id = list(range(0, num_view, 3))
        else:
            self.val_novel_id = self.opt.val_novel_id
        
        self.source_id = self.opt.source_id
        self.side_id = self.opt.side_id
        if self.opt.source_id == 'all':
            self.num_view = int(opt.data_root.split('_')[-1])
            source_id0 = np.random.choice(list(range(self.num_view)))
            source_id1 = (source_id0 + self.num_view // 2) % self.num_view
            side_id0 = (source_id0 + self.num_view // 4) % self.num_view
            side_id1 = (source_id1 + self.num_view // 4) % self.num_view
            self.source_id = [source_id0, source_id1]
            self.side_id = [side_id0, side_id1]


    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True, require_pts=True, half_res=False):
        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        depth_name = self.depth_path % (sample_name, source_id)
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)

        if 2500 > int(sample_name.split('_')[0]) >= 526 and self.opt.add_2_1:
            img_name = img_name.replace('2.0', '2.1')
            image_hr_name = image_hr_name.replace('2.0', '2.1')
            mask_name = mask_name.replace('2.0', '2.1')
            depth_name = depth_name.replace('2.0', '2.1')
            intr_name = intr_name.replace('2.0', '2.1')
            extr_name = extr_name.replace('2.0', '2.1')
            
        if int(sample_name.split('_')[0]) >= 2500 and self.opt.add_ch:
            img_name = img_name.replace('2.0', 'ch')
            image_hr_name = image_hr_name.replace('2.0', 'ch')
            mask_name = mask_name.replace('2.0', 'ch')
            depth_name = depth_name.replace('2.0', 'ch')
            intr_name = intr_name.replace('2.0', 'ch')
            extr_name = extr_name.replace('2.0', 'ch')

        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts = None, None

        if require_mask:
            mask = read_img(mask_name)

        if hr_img:
            img = read_img(image_hr_name)
            intr[:2] *= 2
        else:
            img = read_img(img_name)

        if half_res:
            img = img[::2, ::2]
            intr[:2] /= 2

            if require_mask and not hr_img:
                mask = mask[::2, ::2]
            
            if require_pts:
                if not hr_img:
                    pts = pts[::2, ::2]

        return img, mask, intr, extr


    def get_novel_view_tensor(self, sample_name, view_id, half_res=False):
        img, mask, intr, extr = self.load_single_view(sample_name, view_id, hr_img=self.opt.use_hr_img,
                                                                require_mask=True, require_pts=True, half_res=half_res)
        width, height = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        mask = mask / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': torch.FloatTensor(mask),
            'intr': torch.FloatTensor(intr),
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        return novel_view_data


    def get_rectified_stereo_data_none(self, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0 = main_view_data
        img1, mask1, intr1, extr1 = ref_view_data

        camera = {
            'intr0': intr0,
            'intr1': intr1,
            'extr0': extr0,
            'extr1': extr1,
            'Tf_x': None
        }

        stereo_data = {
            'img0': img0,
            'mask0': mask0,
            'img1': img1,
            'mask1': mask1,
            'camera': camera,
        }

        return stereo_data


    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'idx': 0,
            'instance': 'left',
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'idx': 1,
            'instance': 'right',
        }

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}


    def get_item(self, index, side_id=[4, 9], novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        half_res = False
        
        source_list = [[0, 6], [6, 12], [12, 18], [18, 24], [24, 0]]
        self.source_id = source_list[np.random.randint(0, len(source_list))]

        view0_data = self.load_single_view(sample_name, self.source_id[0], hr_img=False,
                                            require_mask=True, require_pts=True, half_res=half_res)
        view1_data = self.load_single_view(sample_name, self.source_id[1], hr_img=False,
                                            require_mask=True, require_pts=True, half_res=half_res)
        stereo_np = self.get_rectified_stereo_data_none(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)


        dust3r_pred_path = self.dust3r_path % (sample_name[:4], sample_name[:4])
        
        if 2500 > int(sample_name.split('_')[0]) >= 526 and self.opt.add_2_1:
            dust3r_pred_path = dust3r_pred_path.replace('2.0', '2.1')
            
        if int(sample_name.split('_')[0]) >= 2500 and self.opt.add_ch:
            dust3r_pred_path = dust3r_pred_path.replace('2.0', 'ch')
        
        dust3r_pred = torch.load(dust3r_pred_path, map_location='cpu', weights_only=False)
        dict_tensor.update({
            'dust3r_pred': dust3r_pred,
        })


        if novel_id:
            novel_id = np.random.choice(novel_id, 3)
            dict_tensor.update({
                'novel_view1': self.get_novel_view_tensor(sample_name, novel_id[0], half_res=half_res),
                'novel_view2': self.get_novel_view_tensor(sample_name, novel_id[1], half_res=half_res),
                'novel_view3': self.get_novel_view_tensor(sample_name, novel_id[2], half_res=half_res),
                'novel_view4': self.get_novel_view_tensor(sample_name, side_id[0], half_res=half_res),
                'novel_view5': self.get_novel_view_tensor(sample_name, side_id[1], half_res=half_res),
            })

        return dict_tensor


    def get_test_item(self, index, source_id, half_res=False):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        view0_data = self.load_single_view(sample_name, source_id[0], hr_img=False, require_mask=True, require_pts=True, half_res=False)
        view1_data = self.load_single_view(sample_name, source_id[1], hr_img=False, require_mask=True, require_pts=True, half_res=False)
        stereo_np = self.get_rectified_stereo_data_none(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        img_len = 2048 if self.opt.use_hr_img else 1024
        img_len = img_len // 2 if half_res else img_len
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })

        return dict_tensor

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, side_id=self.side_id, novel_id=self.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, side_id=self.side_id, novel_id=self.val_novel_id)

    # def __len__(self):
    #     return len(self.sample_list)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)
