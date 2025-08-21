#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import cv2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(data, idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color, name):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data[name]['FovX'][idx] * 0.5)
    tanfovy = math.tan(data[name]['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data[name]['height'][idx]),
        image_width=int(data[name]['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data[name]['world_view_transform'][idx],
        projmatrix=data[name]['full_proj_transform'][idx],
        sh_degree=3,
        campos=data[name]['camera_center'][idx],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # import pdb;pdb.set_trace()

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=pts_xyz.float(),
        means2D=screenspace_points.float(),
        shs=None,
        colors_precomp=pts_rgb.float(),
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=None)

    return rendered_image, depth, alpha
