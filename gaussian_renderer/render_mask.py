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
from diff_gaussian_rasterization_mask import GaussianRasterizationSettings, GaussianRasterizer 
from scene.gaussian_model import GaussianModel 
from utils.sh_utils import eval_sh

def render_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, gt_mask=None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    all_xyz = pc.get_xyz
    screenspace_points = torch.zeros_like(all_xyz, dtype=all_xyz.dtype, requires_grad=True, device="cuda") + 0
    mask_shunts = torch.zeros((all_xyz.shape[0], 2), dtype=torch.float32, requires_grad=True, device="cuda") + 0
    mask_signals = torch.zeros((all_xyz.shape[0], 2), dtype=torch.float32, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
        mask_shunts.retain_grad()
        mask_signals.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # print(viewpoint_camera.image_height, viewpoint_camera.image_width)
    if gt_mask is None:
        mask_height, mask_width = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    else:
        mask_height, mask_width = gt_mask.shape[0], gt_mask.shape[1]

    raster_settings = GaussianRasterizationSettings(
        image_height=mask_height,
        image_width=mask_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = all_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    mask_precomp = pc.get_identity
    
    if len(mask_precomp.shape) == 1 or mask_precomp.shape[-1] == 1:
        mask_precomp = mask_precomp.squeeze().unsqueeze(-1).repeat([1,3]).cuda()

    shs = None
    colors_precomp = mask_precomp.float()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_mask, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mask_shunts = mask_shunts,
        mask_signals = mask_signals
        )
    
    rendered_mask = rendered_mask[0].clamp(0, 1)  
    
    out = {
            "mask": rendered_mask,
            "mask_shunts": mask_shunts,
            "mask_signals": mask_signals,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth_image": depth_image
        }
    
    return out