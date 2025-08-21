
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from plyfile import PlyData, PlyElement

from .sh_utils import RGB2SH

def get_novel_calib(data, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        # import pdb;pdb.set_trace()
        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


def get_eval_calib(data, opt, sample_name, subangle):
    data_root = os.path.join(opt.data_root)
    intr_path = os.path.join(data_root, 'parm/%s/%d_intrinsic.npy')
    extr_path = os.path.join(data_root, 'parm/%s/%d_extrinsic.npy')

    bs = 1 # TODO
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list, extr_list, intr_list = [], [], [], [], [], [], []
    for i in range(bs):

        width, height = 1024,1024 # TODO
        intr_name = intr_path % (sample_name, subangle)
        extr_name = extr_path % (sample_name, subangle)
        intr, extr = np.load(intr_name), np.load(extr_name)
        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))
        extr_list.append(torch.tensor(extr).unsqueeze(0))
        intr_list.append(torch.tensor(intr).unsqueeze(0))
        
    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    data['novel_view']['extr'] = torch.concat(extr_list).cuda()
    data['novel_view']['intr'] = torch.concat(intr_list).cuda()
    return data


def depth2pc(depth, extrinsic, intrinsic, scale=1, flag=True):
    B, S, S = depth.shape
    depth = depth[:, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    # import pdb;pdb.set_trace()
    # pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 2] = depth * scale
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2] # cu
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2] # cv
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None] # fu
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None] # fv 

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1) # [1, 3, 262144]
    if flag:
        rot_t = rot.permute(0, 2, 1) # [1, 3, 3]
        pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)
    else:
        pts = pts_2d

    return pts.permute(0, 2, 1).reshape(B, 3, S, S) # .reshape(B, 3, S, S)


def pts_cam2world(pts, extrinsic):
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]
    rot_t = rot.permute(0, 2, 1)

    if pts.dim() == 3:
        pts = pts.permute(0, 2, 1)
        pts = torch.bmm(rot_t, pts) - torch.bmm(rot_t, trans)
        pts = pts.permute(0, 2, 1)
    else:
        B, H, W, C = pts.shape
        pts = pts.view(B, -1, C).permute(0, 2, 1)
        pts = torch.bmm(rot_t, pts) - torch.bmm(rot_t, trans)
        pts = pts.permute(0, 2, 1).reshape(B, H, W, C)
    return pts


def pts_transform(pts, extrinsic, need_inv=False):
    homo_extr = torch.cat([extrinsic, torch.tensor([[0,0,0,1]],device=pts.device)], dim=0)
    homo_pts = torch.cat([pts, torch.ones([len(pts), 1],device=pts.device)], dim=1)
    if need_inv:
        homo_extr = homo_extr.T
    final_pts = (homo_extr @ homo_pts.T).T
    return final_pts[:, :-1]


def pts2depth_n3(point_cloud, extrinsics, intrinsics, size=(1024,1024)):
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    point_cloud_cam = (R @ point_cloud.T).T + T

    # valid_mask = point_cloud_cam[:, 2] > 0
    # point_cloud_cam = point_cloud_cam[valid_mask]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = point_cloud_cam[:, 0] / point_cloud_cam[:, 2]
    y = point_cloud_cam[:, 1] / point_cloud_cam[:, 2]
    
    # import pdb;pdb.set_trace()

    uu = (fx * x + cx).long()
    vv = (fy * y + cy).long()

    depth = torch.full(size, float('inf'), device=point_cloud.device)
    depth_mask = (uu >= 0) & (uu < size[0]) & (vv >= 0) & (vv < size[1])

    uu, vv, zz = uu[depth_mask], vv[depth_mask], point_cloud_cam[depth_mask, 2]
    
    depth[vv, uu] = torch.minimum(depth[vv, uu], zz)

    # for ui, vi, zi in zip(uu, vv, zz):
    #     depth[vi, ui] = min(depth[vi, ui], zi)

    depth[depth == float('inf')] = 0

    # cv2
    
    # kernel_size = 3
    # kernel = torch.ones(kernel_size, kernel_size, device=depth.device)  # 3x3 卷积核
    # depth = F.conv2d(depth.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
    # depth = depth.squeeze(0).squeeze(0) / kernel_size**2

    # img_z_shift = torch.stack([
    #     depth,
    #     torch.roll(depth, 1, dims=0),
    #     torch.roll(depth, -1, dims=0),
    #     torch.roll(depth, 1, dims=1),
    #     torch.roll(depth, -1, dims=1)
    # ])
    # depth = torch.min(img_z_shift, dim=0)[0]

    return depth


# def pts2depth(point_cloud, extrinsics, intrinsics, size=1024):
#     R = extrinsics[:3, :3]
#     T = extrinsics[:3, 3]
#     point_cloud_cam = (R @ point_cloud.T).T + T

#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
#     x = point_cloud_cam[:, 0] / point_cloud_cam[:, 2]
#     y = point_cloud_cam[:, 1] / point_cloud_cam[:, 2]
    
#     uu = (fx * x + cx) / size * 2 - 1
#     vv = (fy * y + cy) / size * 2 - 1

#     # import pdb;pdb.set_trace()
#     depth = torch.full((size, size), float('inf'), device=point_cloud.device)
#     depth_mask = (point_cloud_cam[:, 2] > 0)

#     uu, vv, zz = uu[depth_mask], vv[depth_mask], point_cloud_cam[depth_mask, 2]
    
#     # create grid
#     grid = torch.full((size, size, 2), float('inf'), device=point_cloud.device)
#     grid[vv, uu] = torch.stack([uu, vv], dim=-1)

#     depth[vv, uu] = torch.minimum(depth[vv, uu], zz)

#     # for ui, vi, zi in zip(uu, vv, zz):
#     #     depth[vi, ui] = min(depth[vi, ui], zi)

#     depth[depth == float('inf')] = 0

#     # cv2
    
#     # kernel_size = 3
#     # kernel = torch.ones(kernel_size, kernel_size, device=depth.device)  # 3x3 卷积核
#     # depth = F.conv2d(depth.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
#     # depth = depth.squeeze(0).squeeze(0) / kernel_size**2

#     # img_z_shift = torch.stack([
#     #     depth,
#     #     torch.roll(depth, 1, dims=0),
#     #     torch.roll(depth, -1, dims=0),
#     #     torch.roll(depth, 1, dims=1),
#     #     torch.roll(depth, -1, dims=1)
#     # ])
#     # depth = torch.min(img_z_shift, dim=0)[0]

#     return depth


def pts2depth(point_cloud, extrinsics, intrinsics, size=(1024,1024)):
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    point_cloud_cam = (R @ point_cloud.T).T + T

    # valid_mask = point_cloud_cam[:, 2] > 0
    # point_cloud_cam = point_cloud_cam[valid_mask]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = point_cloud_cam[:, 0] / point_cloud_cam[:, 2]
    y = point_cloud_cam[:, 1] / point_cloud_cam[:, 2]
    
    uu = (fx * x + cx).long()
    vv = (fy * y + cy).long()

    depth = torch.full(size, float('inf'), device=point_cloud.device)
    depth_mask = (uu >= 0) & (uu < size[0]) & (vv >= 0) & (vv < size[1])

    uu, vv, zz = uu[depth_mask], vv[depth_mask], point_cloud_cam[depth_mask, 2]
    
    depth[vv, uu] = torch.minimum(depth[vv, uu], zz)
    depth[depth == float('inf')] = 0

    return depth


def pts2depth_hw(point_cloud, extrinsic, intrinsic, size=1024):
    # import pdb;pdb.set_trace()
    intrinsic = torch.from_numpy(intrinsic).float().cuda()
    extrinsic = extrinsic[:3, :4]

    pts = point_cloud.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]

    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = pts[2, :].view(size, size) # [H, W]
    # cv2.imwrite('depth1.png', depth.cpu().detach().numpy() * 255)

    # xy = pts[:2, :].view(2, size, size).permute(1, 2, 0) # [H, W, 2]
    # xy_norm = xy / ((size - 1) / 2) - 1
    # depth_sampled = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), xy_norm.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)
    # cv2.imwrite('depth.png', depth_sampled[0][0].cpu().detach().numpy() * 255)

    return depth


# def pts2rgb(point_cloud, color, extrinsic, intrinsic, size=1024):
#     intrinsic = torch.from_numpy(intrinsic).float().cuda()
#     extrinsic = extrinsic[:3, :4]

#     pts = point_cloud.view(-1, 3).T
#     calib = intrinsic @ extrinsic
#     pts = calib[:3, :3] @ pts
#     pts = pts + calib[:3, 3:4]
    
#     pts[:2, :] /= (pts[2:, :] + 1e-8)
#     x_norm = pts[0, :] / ((size -1) / 2) - 1
#     y_norm = pts[1, :] / ((size -1) / 2) - 1
#     proj_xy = torch.stack([x_norm, y_norm], dim=-1)

#     rgb_image = F.grid_sample(color.unsqueeze(0), proj_xy.view(1, size, size, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
#     rgb_image = rgb_image.view(3, size, size)

#     return rgb_image



def differentiable_warping(src_fea, src_proj, ref_proj, depth_samples):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, Ndepth, H, W] 
    # out: [B, C, Ndepth, H, W]
    batch, channels, height, width = src_fea.shape
    num_depth = depth_samples.shape[1]
    
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                            height * width)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = width
        proj_xyz[:, 1:2][negative_depth_mask] = height
        proj_xyz[:, 2:3][negative_depth_mask] = 1
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1 # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy      

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                padding_mode='zeros',align_corners=True)
    
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


import torch.nn.functional as F
def get_rays(pose, h, w, focal):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = (rays_d / torch.norm(rays_d, dim=-1, keepdim=True)).view(h, w, 3)
    return rays_o, rays_d


def pts2rgb(point_cloud, colors, extrinsics, intrinsics, size=1024):
    # 提取旋转矩阵和平移向量
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    
    # 将点云转换到相机坐标系
    point_cloud_cam = (R @ point_cloud.T).T + T

    # 相机内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 计算归一化的图像平面坐标
    x = point_cloud_cam[:, 0] / point_cloud_cam[:, 2]
    y = point_cloud_cam[:, 1] / point_cloud_cam[:, 2]

    # 像素坐标 (u, v)
    uu = (fx * x + cx).long()
    vv = (fy * y + cy).long()

    # 初始化深度图和 RGB 图像
    depth = torch.full((size,size), float('inf'), device=point_cloud.device)
    rgb_image = torch.zeros((size, size, 3), device=point_cloud.device) - 1
    # rgb_image = torch.zeros((size, size, 3), device=point_cloud.device) + 1

    # 只保留有效像素（在图像范围内且深度为正）
    depth_mask = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)

    uu, vv = uu[depth_mask], vv[depth_mask]
    zz = point_cloud_cam[depth_mask, 2]
    valid_colors = colors[depth_mask]

    # 使用深度进行可见性检查，更新 RGB 图像
    depth_indices = (zz < depth[vv, uu])
    depth[vv[depth_indices], uu[depth_indices]] = zz[depth_indices]
    rgb_image[vv[depth_indices], uu[depth_indices]] = valid_colors[depth_indices]
    
    # rgb_image[vv[depth_indices], uu[depth_indices]] = torch.tensor([1., 1., 1.], device=point_cloud.device)
    # import pdb;pdb.set_trace()
    # cv2.imwrite('rgb.png', (rgb_image.cpu().detach().numpy()+1)/2 * 255)

    # grid = torch.full((size, size, 2), 0., device=point_cloud.device)
    # grid[vv, uu] = torch.stack([(fx * x + cx), (fy * y + cy)], dim=-1)
    # grid = grid / size * 2 - 1

    # rgb_image_new = F.grid_sample(rgb_image.permute(2, 0, 1).unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)
    # cv2.imwrite('rgb_new.png', (rgb_image_new[0].permute(1, 2, 0).cpu().detach().numpy()+1)/2 * 255)

    return rgb_image


def depth2pts(depth, extrinsics, intrinsics):
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 创建网格坐标
    y, x = torch.meshgrid(torch.arange(W, device=depth.device), torch.arange(H, device=depth.device))
    
    # 计算点云坐标
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # 堆叠成点云
    point_cloud = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    # 应用外参
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    point_cloud_world = (R.T @ (point_cloud.T - T.unsqueeze(1))).T

    return point_cloud_world.reshape(H, W, 3)


def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'bad {focal_mode=}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


def show_extr(focal, pp, extr, W=1024, H=1024):
    intr = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-intr[0][2])/intr[0][0], (j-intr[1][2])/intr[1][1], np.ones_like(i)], -1)[::8, ::8, :]
    coord_points = torch.tensor(dirs, dtype=torch.float32).view(-1, 3).cuda()
    coord_points = torch.cat([coord_points, torch.tensor([[0, 0, 0]], dtype=torch.float32).cuda()], dim=0)
    coord_points = pts_cam2world(coord_points.unsqueeze(0), extr.unsqueeze(0))
    return coord_points.squeeze(0).cpu().detach().numpy()


def gen_c2w_from_theta(theta, radius, phi=0):
    c2w = rot_theta(theta/180.*np.pi) @ rot_phi(phi/180.*np.pi) @ trans_t(radius)
    c2w = torch.Tensor(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,float(radius)],[0,0,0,1]])) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])) @ inv(c2w)
    return c2w


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,radius,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d+1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim-2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def transform_extrinsics(A, B):
    
    R_A, T_A = A[:3, :3], A[:3, 3]
    R_B, T_B = B[:3, :3], B[:3, 3]

    R_B_new = R_B @ R_A.T
    T_B_new = (T_B - T_A.unsqueeze(1).T @ R_A @ R_B.T).T 

    B_new = torch.cat([R_B_new, T_B_new], dim=1)    
    return B_new


# def depth2pts(depth, extrinsic, intrinsic):
#     # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
#     rot = extrinsic[:3, :3]
#     trans = extrinsic[:3, 3:]
#     S, S = depth.shape

#     y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
#                           torch.linspace(0.5, S-0.5, S, device=depth.device))
#     pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

#     pts_2d[..., 2] = depth
#     pts_2d[..., 0] -= intrinsic[0, 2]
#     pts_2d[..., 1] -= intrinsic[1, 2]
#     pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
#     pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

#     pts_2d[..., 0] /= intrinsic[0, 0]
#     pts_2d[..., 1] /= intrinsic[1, 1]

#     pts_2d = pts_2d.reshape(-1, 3).T
#     pts = rot.T @ pts_2d - rot.T @ trans
#     return pts.T.view(S, S, 3)


def flow2depth(data):
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    offset = torch.broadcast_to(offset[:, None, None, None], data['flow_pred'].shape)
    disparity = offset - data['flow_pred']
    depth = -disparity / data['Tf_x'][:, None, None, None]
    depth *= data['mask'][:, :1, :, :]

    return depth


def save_pts(xyz, path, color='black'):
    from plyfile import PlyData, PlyElement

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    def gen_ply(xyz, color):
        normals = np.zeros_like(xyz)
        if color == 'black':
            rgb = np.zeros_like(xyz)
        elif color == 'white':
            rgb = np.ones_like(xyz) * 255
        elif color == 'red':
            rgb = np.ones_like(xyz) * [255, 0, 0]
        elif color == 'blue':
            rgb = np.ones_like(xyz) * [0, 0, 255]
        elif color == 'green':
            rgb = np.ones_like(xyz) * [0, 255, 0]

        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        return attributes

    if isinstance(xyz, np.ndarray):
        attributes = gen_ply(xyz, color)
    else:
        attributes = []
        for i in range(len(xyz)):
            attributes.append(gen_ply(xyz[i], color[i]))
        attributes = np.concatenate(attributes, axis=0)

    elements = np.empty(attributes.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def save_pts_with_colors(xyz, path, color):
    from plyfile import PlyData, PlyElement

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, color), axis=1)
    elements = np.empty(attributes.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


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