
import os
import cv2
import torch
import logging
import numpy as np
from gaussian_renderer import render
from plyfile import PlyData, PlyElement
from lib.utils import *
from .sh_utils import RGB2SH


def pts2render(data, bg_color, count=-1):
    bs = data['lmain']['img'].shape[0]
    render_novel_list, render_depth_list, render_alpha_list = [], [], []

    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            # rgb_i = data[view]['rgb_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        # import pdb;pdb.set_trace()
        # save_gaussians_as_ply('debug/sample.ply', pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i)
        # if count >= 0 and count % 100 == 0:
        #     save_gaussians_as_ply('debug/{}.ply'.format(count), pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i)
        # import pdb;pdb.set_trace()

        render_novel_i, render_depth_i, render_alpha_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        # cv2.imwrite('debug_/{}.jpg'.format(i), render_novel_i.permute(1,2,0).cpu().detach().numpy()[...,::-1]*255)
        # import pdb;pdb.set_trace()
        render_novel_list.append(render_novel_i.unsqueeze(0))
        render_depth_list.append(render_depth_i)
        render_alpha_list.append(render_alpha_i)

    # import pdb;pdb.set_trace()

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    data['novel_view']['depth_pred'] = torch.concat(render_depth_list, dim=0)
    data['novel_view']['alpha_pred'] = torch.concat(render_alpha_list, dim=0)
    return data


def pts2render_human(data, bg_color, mode='lrss', name='novel_view'):
    bs = data['lmain']['img'].shape[0]
    render_novel_list, render_depth_list, render_alpha_list = [], [], []

    for i in range(bs):

        if mode == 'lrss':  # lr + ss
            valid_i = data['pred'][i]['mask'].clone()
        elif mode == 'lr':  # ss
            valid_i = data['pred'][i]['mask'].clone()
            valid_i[2:, ...] = False
        elif mode == 'ss':  # lr
            valid_i = data['pred'][i]['mask'].clone()
            valid_i[:2, ...] = False
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        # import pdb;pdb.set_trace()
        xyz_i = pts_cam2world(data['pred'][i]['xyz'][valid_i].unsqueeze(0), data['lmain']['extr'][i].unsqueeze(0)).squeeze(0)

        rgb_i = data['pred'][i]['rgb'][valid_i] * 0.5 + 0.5
        rot_i = data['pred'][i]['rot'][valid_i]
        scale_i = data['pred'][i]['scale'][valid_i]
        opacity_i = data['pred'][i]['opacity'][valid_i]

        # save_gaussians_as_ply('debug/sample_gs.ply', xyz_i, rgb_i, rot_i, scale_i, opacity_i)
        # import pdb;pdb.set_trace()

        render_novel_i, render_depth_i, render_alpha_i = render(data, i, xyz_i, rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color, name=name)
        # cv2.imwrite('debug/{}_.jpg'.format(i), render_novel_i.permute(1,2,0).cpu().detach().numpy()[...,::-1]*255)
        # cv2.imwrite('debug/{}_mask.jpg'.format(i), valid_i.cpu().detach().numpy()[...,::-1]*255)
        # import pdb;pdb.set_trace()

        render_novel_list.append(render_novel_i.unsqueeze(0))
        render_depth_list.append(render_depth_i)
        render_alpha_list.append(render_alpha_i)

    data[name]['img_pred'] = torch.concat(render_novel_list, dim=0)
    data[name]['depth_pred'] = torch.concat(render_depth_list, dim=0)
    data[name]['alpha_pred'] = torch.concat(render_alpha_list, dim=0)
    return data


def gather_pts(data):
    i = 0

    valid_i = data['pred'][i]['mask']
    xyz_i = pts_cam2world(data['pred'][i]['xyz'][valid_i].unsqueeze(0), data['lmain']['extr'].unsqueeze(0)).squeeze(0)
    rgb_i = data['pred'][i]['rgb'][valid_i] * 0.5 + 0.5
    rot_i = data['pred'][i]['rot'][valid_i]
    scale_i = data['pred'][i]['scale'][valid_i]
    opacity_i = data['pred'][i]['opacity'][valid_i]

    save_gaussians_as_ply('debug/sample.ply', xyz_i, rgb_i, rot_i, scale_i, opacity_i)
    logging.info('Gathered points saved to debug/sample.ply')


def gather_pts_human(data):
    bs = data['lmain']['img'].shape[0]
    render_novel_list, render_depth_list, render_alpha_list = [], [], []

    for i in range(bs):

        # xyz_i = data['pred']['xyz'][i].view(-1, 3)
        # rgb_i = data['pred']['rgb'][i].view(-1, 3) * 0.5 + 0.5
        # rot_i = data['pred']['rot'][i].view(-1, 4)
        # scale_i = data['pred']['scale'][i].view(-1, 3)
        # opacity_i = data['pred']['opacity'][i].view(-1, 1)

        valid_i = data['pred']['valid'][i].squeeze(-1) > 3
        xyz_i = data['pred']['xyz'][i, ...][valid_i]
        rgb_i = data['pred']['rgb'][i, ...][valid_i] * 0.5 + 0.5
        rot_i = data['pred']['rot'][i, ...][valid_i]
        scale_i = data['pred']['scale'][i, ...][valid_i]
        opacity_i = data['pred']['opacity'][i, ...][valid_i]

    save_gaussians_as_ply('debug/sample.ply', xyz_i, rgb_i, rot_i, scale_i, opacity_i)
    logging.info('Gathered points saved to debug/sample.ply')

    return data


def save_gaussians_as_ply(path, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    # import pdb;pdb.set_trace()

    xyz = pts_xyz_i.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    # fused_color = RGB2SH(pts_rgb_i.detach()[:, [2, 1, 0]])
    fused_color = RGB2SH(pts_rgb_i.detach())

    features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2))
    features[:, :3, 0] = fused_color
    features_dc = features[:, :, 0:1].transpose(1, 2)
    features_rest = features[:, :, 1:].transpose(1, 2)
    f_dc = features_dc.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    f_rest = features_rest.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    opacities = inverse_sigmoid(opacity_i.detach()).cpu().numpy()
    scale = torch.log(scale_i.detach()).cpu().numpy()
    rotation = rot_i.detach().cpu().numpy()

    # features_dc = np.zeros_like(features_dc) # !!
    # features_rest = np.zeros_like(features_rest)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scale, rotation)]

    elements = np.empty(xyz.shape[0], dtype = dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis = 1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    
def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l