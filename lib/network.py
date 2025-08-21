
import cv2
import torch
from torch import nn
import numpy as np
from scipy.spatial import KDTree
import torch.nn.functional as F

from lib.utils import *

from lib.unet.unet import *
from lib.loss import l1_loss, ssim
from torch.cuda.amp import autocast as autocast
# from simple_knn._C import distCUDA2

from dust3r.model import AsymmetricCroCo3DStereo, inf
from mast3r.model import AsymmetricMASt3R
from mast3r.losses import ConfLoss, Regr3D
from dust3r.losses import L21

from pykeops.torch import LazyTensor


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, dust3r=False, inpainting=False, pred=False):
        super().__init__()
        self.cfg = cfg
        # import pdb;pdb.set_trace()

        if dust3r:
            self.dust3r = AsymmetricCroCo3DStereo(
                pos_embed='RoPE100', 
                patch_embed_cls='ManyAR_PatchEmbed', 
                img_size=(512, 512), 
                head_type='dpt', 
                output_mode='pts3d', 
                depth_mode=('exp', -inf, inf), 
                conf_mode=('sigmoid', 0, 1), 
                # conf_mode=('exp', 1, inf), 
                freeze='mask',
                enc_embed_dim=1024, 
                enc_depth=24, 
                enc_num_heads=16, 
                dec_embed_dim=768, 
                dec_depth=12, 
                dec_num_heads=12
            )
            del self.dust3r.mask_token
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
            if self.cfg.dust3r_ckpt is not None:
                self.load_dust3r_ckpt(self.cfg.dust3r_ckpt)
            else:
                model_name = "mast3r/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
                ckpt = torch.load(model_name, weights_only=False)
                self.scale = nn.Parameter(torch.tensor(5.0))
                self.dust3r.load_state_dict(ckpt['model'], strict=False)
                del ckpt

        if pred:
            out_ch = 14
            self.pred_unet = UNet(
                in_channels=6, out_channels=out_ch,
                down_channels=(16, 16, 16),
                down_attention=(False, False, False),
                mid_attention=False,
                up_channels=(16, 16, 16),
                up_attention=(False, False, False),
            )
            self.gs_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)
            if self.cfg.pred_ckpt is not None:
                self.load_pred_ckpt(self.cfg.pred_ckpt)
            
            self.scale_act = lambda x: 0.1 * F.softplus(x, beta=100) # 4
            self.opacity_act = lambda x: torch.sigmoid(x) # 1
            self.rot_act = lambda x: F.normalize(x, dim=-1) # 3
            self.rgb_act = lambda x: torch.tanh(x) # 3
            self.xyz_act = lambda x: 0.001 * x # 3

        self.threshold = 0.9


    def load_dust3r_ckpt(self, load_path):
        assert os.path.exists(load_path)
        ckpt = torch.load(load_path, weights_only=False)
        self.scale = ckpt['network'].pop('scale')
        self.scale.requires_grad_(False)
        
        for key in list(ckpt['network'].keys()):
            ckpt['network'][key.replace('dust3r.', '')] = ckpt['network'].pop(key)
        self.dust3r.load_state_dict(ckpt['network'])
        for param in self.dust3r.parameters():
            param.requires_grad = False
        

    def load_pred_ckpt(self, load_path):
        assert os.path.exists(load_path)
        ckpt = torch.load(load_path, weights_only=False)
        
        temp_dict = {}
        for key in list(ckpt['network'].keys()):
            if 'pred_unet.' in key:
                temp_dict[key.replace('pred_unet.', '')] = ckpt['network'][key]
        self.pred_unet.load_state_dict(temp_dict)
        for param in self.pred_unet.parameters():
            param.requires_grad = False
            
        temp_dict = {}
        for key in list(ckpt['network'].keys()):
            if 'gs_conv.' in key:
                temp_dict[key.replace('gs_conv.', '')] = ckpt['network'][key]
        self.gs_conv.load_state_dict(temp_dict)
        for param in self.gs_conv.parameters():
            param.requires_grad = False


    def forward_side(self, data, train_dust3r=True, add_inpainting_loss=False):

        res1, res2 = data['dust3r_pred']['res1'], data['dust3r_pred']['res2']
        
        front_input = torch.cat([data['lmain']['img'], res1['pts3d'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        back_input = torch.cat([data['rmain']['img'], res2['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)

        pred_feat_fb = self.pred_unet(torch.cat([front_input, back_input], dim=0))
        pred_feat_fb = self.gs_conv(pred_feat_fb).permute(0, 2, 3, 1)
        
        pred_feat = pred_feat_fb
    
        data['pred'], pred_human = [], {}
        pred_human['xyz'] = torch.cat([res1['pts3d'].squeeze(1), res2['pts3d_in_other_view'].squeeze(1)], dim=0).cuda()
        pred_human['xyz'] += self.xyz_act(pred_feat[..., 11:14])
        pred_human['opacity'] = self.opacity_act(pred_feat[..., :1])
        pred_human['scale'] = self.scale_act(pred_feat[..., 1:4])
        pred_human['rot'] = self.rot_act(pred_feat[..., 4:8])
        pred_human['rgb'] = self.rgb_act(pred_feat[..., 8:11])

        pred_human['mask'] = torch.cat([
            data['lmain']['mask'][:, 0] > 0., 
            data['rmain']['mask'][:, 0] > 0., 
        ], dim=0).cuda()
        data['pred'].append(pred_human)

        return data


    def forward_demo_wo_side(self, data):
        l_img = data['lmain']['img']
        r_img = data['rmain']['img']

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))
        img_size = 512

        view1, view2 = data['lmain'], data['rmain']
        res1, res2 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1))
        res2['conf'] = self.up(res2['conf'].unsqueeze(1))
        
        res1['pts3d'] = res1['pts3d'].unsqueeze(1)
        res2['pts3d_in_other_view'] = res2['pts3d_in_other_view'].unsqueeze(1)
        
        front_input = torch.cat([l_img, res1['pts3d'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        back_input = torch.cat([r_img, res2['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)

        pred_feat_fb = self.pred_unet(torch.cat([front_input, back_input], dim=0))
        pred_feat_fb = self.gs_conv(pred_feat_fb).permute(0, 2, 3, 1)
        
        pred_feat = pred_feat_fb
    
        data['pred'], pred_human = [], {}
        pred_human['xyz'] = torch.cat([res1['pts3d'].squeeze(1), res2['pts3d_in_other_view'].squeeze(1)], dim=0).cuda()
        pred_human['xyz'] += self.xyz_act(pred_feat[..., 11:14])
        pred_human['opacity'] = self.opacity_act(pred_feat[..., :1])
        pred_human['scale'] = self.scale_act(pred_feat[..., 1:4])
        pred_human['rot'] = self.rot_act(pred_feat[..., 4:8])
        pred_human['rgb'] = self.rgb_act(pred_feat[..., 8:11])
        
        pred_human['mask'] = torch.cat([
            data['lmain']['mask'][:1] > 0., 
            data['rmain']['mask'][:1] > 0., 
        ], dim=0).cuda()
        data['pred'].append(pred_human)

        return data


    def forward_demo(self, data):
        l_img = data['lmain']['img']
        r_img = data['rmain']['img']

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))

        view1, view2 = data['lmain'], data['rmain']
        res1, res2, res3, res4 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res3['pts3d_in_other_view'] = self.up(res3['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res4['pts3d_in_other_view'] = self.up(res4['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1))
        res2['conf'] = self.up(res2['conf'].unsqueeze(1))
        res3['conf'] = self.up(res3['conf'].unsqueeze(1))
        res4['conf'] = self.up(res4['conf'].unsqueeze(1))
        
        res1['pts3d'] = res1['pts3d'].unsqueeze(1)
        res2['pts3d_in_other_view'] = res2['pts3d_in_other_view'].unsqueeze(1)
        res3['pts3d_in_other_view'] = res3['pts3d_in_other_view'].unsqueeze(1)
        res4['pts3d_in_other_view'] = res4['pts3d_in_other_view'].unsqueeze(1)

        valid1, valid2 = view1['mask'][0].unsqueeze(0).unsqueeze(0) >= 1., view2['mask'][0].unsqueeze(0).unsqueeze(0) >= 1.
        # valid1, valid2 = view1['mask'].unsqueeze(0).unsqueeze(0) >= 1., view2['mask'].unsqueeze(0).unsqueeze(0) >= 1.
        valid3, valid4 = res3['conf'] > self.threshold, res4['conf'] > self.threshold

        all_lr_pts = torch.cat([res1['pts3d'][valid1], res2['pts3d_in_other_view'][valid2]], dim=0).cuda()
        all_lr_colors = torch.cat([l_img.permute(0, 2, 3, 1)[valid1.squeeze(1)], r_img.permute(0, 2, 3, 1)[valid2.squeeze(1)]], dim=0).cuda()
        
        right_pts, left_pts = res3['pts3d_in_other_view'][valid3], res4['pts3d_in_other_view'][valid4]

        x_i = LazyTensor(right_pts[:, None, :])
        y_j = LazyTensor(all_lr_pts[None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        _, nearest_idx = D_ij.Kmin_argKmin(1, dim=1)
        right_colors = all_lr_colors[nearest_idx.squeeze(1)]
        right_img = torch.zeros([1, 1024, 1024, 3]).cuda() - 1
        right_img[valid3.squeeze(1)] = right_colors
        right_img = right_img.permute(0, 3, 1, 2)
        
        x_i = LazyTensor(left_pts[:, None, :])
        y_j = LazyTensor(all_lr_pts[None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        _, nearest_idx = D_ij.Kmin_argKmin(1, dim=1)
        left_colors = all_lr_colors[nearest_idx.squeeze(1)]
        left_img = torch.zeros([1, 1024, 1024, 3]).cuda() - 1
        left_img[valid4.squeeze(1)] = left_colors
        left_img = left_img.permute(0, 3, 1, 2)
        
        right_input = torch.cat([right_img, res3['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        left_input = torch.cat([left_img, res4['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        
        pred_feat_side = self.pred_unet(torch.cat([right_input, left_input], dim=0))
        pred_feat_side = self.gs_conv(pred_feat_side).permute(0, 2, 3, 1)
        
        front_input = torch.cat([l_img, res1['pts3d'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        back_input = torch.cat([r_img, res2['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)

        pred_feat_fb = self.pred_unet(torch.cat([front_input, back_input], dim=0))
        pred_feat_fb = self.gs_conv(pred_feat_fb).permute(0, 2, 3, 1)
        
        pred_feat = torch.cat([pred_feat_fb, pred_feat_side], dim=0)

        data['pred'], pred_human = [], {}
        pred_human['xyz'] = torch.cat([
            res1['pts3d'].squeeze(1), 
            res2['pts3d_in_other_view'].squeeze(1),
            res3['pts3d_in_other_view'].squeeze(1), 
            res4['pts3d_in_other_view'].squeeze(1)
        ], dim=0).cuda()
        pred_human['xyz'] += self.xyz_act(pred_feat[..., 11:14])
        pred_human['opacity'] = self.opacity_act(pred_feat[..., :1])
        pred_human['scale'] = self.scale_act(pred_feat[..., 1:4])
        pred_human['rot'] = self.rot_act(pred_feat[..., 4:8])
        pred_human['rgb'] = self.rgb_act(pred_feat[..., 8:11])
        pred_human['mask'] = torch.cat([
            valid1[0], 
            valid2[0], 
            res3['conf'].squeeze(1) > self.threshold, 
            res4['conf'].squeeze(1) > self.threshold
        ], dim=0).cuda()
        data['pred'].append(pred_human)

        return data 


    def forward_kdtree(self, data):
        res1, res2, res3, res4 = data['dust3r_pred']['res1'], data['dust3r_pred']['res2'], data['dust3r_pred']['res3'], data['dust3r_pred']['res4']

        right_img = res3['img'].squeeze(1).cuda()
        left_img = res4['img'].squeeze(1).cuda()

        right_input = torch.cat([right_img, res3['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        left_input = torch.cat([left_img, res4['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)

        pred_feat_side = self.pred_unet(torch.cat([right_input, left_input], dim=0))
        pred_feat_side = self.gs_conv(pred_feat_side).permute(0, 2, 3, 1)
        
        front_input = torch.cat([data['lmain']['img'], res1['pts3d'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)
        back_input = torch.cat([data['rmain']['img'], res2['pts3d_in_other_view'].squeeze(1).permute(0, 3, 1, 2).cuda()], dim=1)

        pred_feat_fb = self.pred_unet(torch.cat([front_input, back_input], dim=0))
        pred_feat_fb = self.gs_conv(pred_feat_fb).permute(0, 2, 3, 1)
        
        pred_feat = torch.cat([pred_feat_fb, pred_feat_side], dim=0)
    
        data['pred'], pred_human = [], {}
        pred_human['xyz'] = torch.cat([res1['pts3d'].squeeze(1), res2['pts3d_in_other_view'].squeeze(1), res3['pts3d_in_other_view'].squeeze(1), res4['pts3d_in_other_view'].squeeze(1)], dim=0).cuda()
        pred_human['xyz'] += self.xyz_act(pred_feat[..., 11:14])
        pred_human['opacity'] = self.opacity_act(pred_feat[..., :1])
        pred_human['scale'] = self.scale_act(pred_feat[..., 1:4])
        pred_human['rot'] = self.rot_act(pred_feat[..., 4:8])
        pred_human['rgb'] = self.rgb_act(pred_feat[..., 8:11])

        pred_human['mask'] = torch.cat([
            data['lmain']['mask'][:, 0] > 0., 
            data['rmain']['mask'][:, 0] > 0., 
            res3['conf'].squeeze(1) > self.threshold, 
            res4['conf'].squeeze(1) > self.threshold
        ], dim=0).cuda()
        data['pred'].append(pred_human)
        return data


    def forward_dust3r(self, data):

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))
        img_size = 512

        # left-right only
        view1, view2 = data['lmain'], data['rmain']
        res1, res2, res3, res4 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res3['pts3d_in_other_view'] = self.up(res3['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res4['pts3d_in_other_view'] = self.up(res4['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1)).squeeze(1)
        res2['conf'] = self.up(res2['conf'].unsqueeze(1)).squeeze(1)
        res3['conf'] = self.up(res3['conf'].unsqueeze(1)).squeeze(1)
        res4['conf'] = self.up(res4['conf'].unsqueeze(1)).squeeze(1)

        loss = {}
        if True:
            gt_pts1 = geotrf(view1['extr'], view1['pts3d'])  # in view 1
            gt_pts2 = geotrf(view1['extr'], view2['pts3d'])
            gt_pts3 = geotrf(view1['extr'], data['side_view1']['pts3d'].cuda())
            gt_pts4 = geotrf(view1['extr'], data['side_view2']['pts3d'].cuda())

            valid1 = view1['mask'][:, 0].bool()
            valid2 = view2['mask'][:, 0].bool()
            valid3 = data['side_view1']['mask'][..., 0].cuda().bool()
            valid4 = data['side_view2']['mask'][..., 0].cuda().bool()

            pr_pts1, pr_pts2 = res1['pts3d'], res2['pts3d_in_other_view']
            pr_pts3, pr_pts4 = res3['pts3d_in_other_view'], res4['pts3d_in_other_view']

            loss_reg = torch.norm(gt_pts1[valid1] - pr_pts1[valid1], dim=-1).mean() + \
                       torch.norm(gt_pts2[valid2] - pr_pts2[valid2], dim=-1).mean() + \
                       torch.norm(gt_pts3[valid3] - pr_pts3[valid3], dim=-1).mean() + \
                       torch.norm(gt_pts4[valid4] - pr_pts4[valid4], dim=-1).mean()
            
            loss_conf = F.binary_cross_entropy(res1['conf'], valid1.float()) + \
                        F.binary_cross_entropy(res2['conf'], valid2.float()) + \
                        F.binary_cross_entropy(res3['conf'], valid3.float()) + \
                        F.binary_cross_entropy(res4['conf'], valid4.float())

            loss['reg'] = loss_reg
            loss['conf'] = loss_conf

        return data, loss
    
    
    def forward_dust3r_wo_side(self, data):

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))
        img_size = 512

        view1, view2 = data['lmain'], data['rmain']
        res1, res2 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1)).squeeze(1)
        res2['conf'] = self.up(res2['conf'].unsqueeze(1)).squeeze(1)

        loss = {}
        if True:
            gt_pts1 = geotrf(view1['extr'], view1['pts3d'])  # in view 1
            gt_pts2 = geotrf(view1['extr'], view2['pts3d'])

            valid1 = view1['mask'][:, 0].bool()
            valid2 = view2['mask'][:, 0].bool()

            pr_pts1, pr_pts2 = res1['pts3d'], res2['pts3d_in_other_view']

            loss_reg = torch.norm(gt_pts1[valid1] - pr_pts1[valid1], dim=-1).mean() + \
                       torch.norm(gt_pts2[valid2] - pr_pts2[valid2], dim=-1).mean() 
            
            loss_conf = F.binary_cross_entropy(res1['conf'], valid1.float()) + \
                        F.binary_cross_entropy(res2['conf'], valid2.float()) 

            loss['reg'] = loss_reg
            loss['conf'] = loss_conf
        return data, loss

    
    def infer_dust3r(self, data):
        l_img = data['lmain']['img']
        r_img = data['rmain']['img']

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))

        # left-right only
        view1, view2 = data['lmain'], data['rmain']
        res1, res2, res3, res4 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res3['pts3d_in_other_view'] = self.up(res3['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res4['pts3d_in_other_view'] = self.up(res4['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1)).squeeze(1)
        res2['conf'] = self.up(res2['conf'].unsqueeze(1)).squeeze(1)
        res3['conf'] = self.up(res3['conf'].unsqueeze(1)).squeeze(1)
        res4['conf'] = self.up(res4['conf'].unsqueeze(1)).squeeze(1)

        valid1, valid2 = view1['mask'][:, 0] >= 1., view2['mask'][:, 0] >= 1.
        valid3, valid4 = res3['conf'] > self.threshold, res4['conf'] > self.threshold

        all_lr_pts = torch.cat([res1['pts3d'][valid1], res2['pts3d_in_other_view'][valid2]], dim=0).cuda()
        all_lr_colors = torch.cat([l_img.permute(0, 2, 3, 1)[valid1], r_img.permute(0, 2, 3, 1)[valid2]], dim=0).cuda()
        
        right_pts, left_pts = res3['pts3d_in_other_view'][valid3], res4['pts3d_in_other_view'][valid4]
        kdtree = KDTree(all_lr_pts.cpu().numpy())
                
        _, nearest_idx = kdtree.query(right_pts.cpu().numpy(), k=1)
        right_colors = all_lr_colors[nearest_idx]
        right_img = torch.zeros([1, 1024, 1024, 3]).cuda() - 1
        right_img[valid3.squeeze(1)] = right_colors
        res3['img'] = right_img.permute(0, 3, 1, 2)
        
        _, nearest_idx = kdtree.query(left_pts.cpu().numpy(), k=1)
        left_colors = all_lr_colors[nearest_idx]
        left_img = torch.zeros([1, 1024, 1024, 3]).cuda() - 1
        left_img[valid4.squeeze(1)] = left_colors
        res4['img'] = left_img.permute(0, 3, 1, 2)

        for x in [res1, res2, res3, res4]:
            for k, v in x.items():
                x[k] = v.cpu()
        
        dust3r_res = {'res1': res1, 'res2': res2, 'res3': res3, 'res4': res4}
        save_path = self.cfg.dataset.data_root + '/human_ply_21_ch/%s'
        if not os.path.exists(save_path % data['name'][0][:4]):
            os.makedirs(save_path % data['name'][0][:4])
        torch.save(dust3r_res, save_path % data['name'][0][:4] + '/%s_dust3r.pth' % data['name'][0][:4])
        print(save_path % data['name'][0][:4])

        return data


    def infer_dust3r_wo_side(self, data):
        l_img = data['lmain']['img']
        r_img = data['rmain']['img']

        data['lmain']['img'] = F.interpolate(data['lmain']['img'], (512, 512))
        data['rmain']['img'] = F.interpolate(data['rmain']['img'], (512, 512))

        # left-right only
        view1, view2 = data['lmain'], data['rmain']
        res1, res2 = self.dust3r(view1=view1, view2=view2)
        
        res1['pts3d'] = self.up(res1['pts3d'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res2['pts3d_in_other_view'] = self.up(res2['pts3d_in_other_view'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * self.scale
        res1['conf'] = self.up(res1['conf'].unsqueeze(1)).squeeze(1)
        res2['conf'] = self.up(res2['conf'].unsqueeze(1)).squeeze(1)

        for x in [res1, res2]:
            for k, v in x.items():
                x[k] = v.cpu()

        dust3r_res = {'res1': res1, 'res2': res2}
        save_path = self.cfg.dataset.data_root + '/human_ply_wo_side/%s'
        if not os.path.exists(save_path % data['name'][0][:4]):
            os.makedirs(save_path % data['name'][0][:4])
        torch.save(dust3r_res, save_path % data['name'][0][:4] + '/%s_dust3r.pth' % data['name'][0][:4])
        print(save_path % data['name'][0][:4])

        return data