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
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
import time
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._identity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._identity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._identity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args

            old_ckpt_without_identity = False

        elif len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args

            old_ckpt_without_identity = True

            self._identity = nn.Parameter(
                torch.zeros(
                    (self._xyz.shape[0], 1),
                    dtype=self._xyz.dtype,
                    device=self._xyz.device
                ).requires_grad_(True)
            )

        else:
            raise ValueError(f"Unsupported checkpoint format: {len(model_args)} items")

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

        if old_ckpt_without_identity:
            opt_dict["param_groups"].append(
                self.optimizer.state_dict()["param_groups"][-1]
            )

        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._identity], 'lr': getattr(training_args, "identify_lr", 0.0), "name": "identity"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._identity = optimizable_tensors["identity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_identity, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "identity" : new_identity}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._identity = optimizable_tensors["identity"]
        if self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") 
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_identity = self._identity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_identity, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_identity = self._identity[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_identity, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    #__________________________________________________________________________________________________________________________________
    def Initialize_identity(self): 
        n = self.get_xyz.shape[0]
        identity = torch.zeros((n, 1), dtype=torch.float32, device="cuda") + 0.5
        
        self._identity = nn.Parameter(identity.requires_grad_(True)) 
   
    def load_identity(self, path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        # 读取 NumPy 数组
        identity = torch.tensor(np.load(path), device="cuda")
        self._identity = nn.Parameter(identity.requires_grad_(True))
    
    def save_identity(self, path):
        if not path:
            raise ValueError("Invalid path: path cannot be empty")       
        os.makedirs(os.path.dirname(path), exist_ok=True)
        identity = self.get_identity.detach().cpu().numpy()
        np.save(path, identity)  # 以 .npy 
    
    @property
    def get_identity(self):
        return torch.sigmoid(self._identity)
    #__________________________________________________________________________________________________________________________________________
    def reset_trends(self):
        self.trend_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")

    def add_densification_stats_trends(self, mask_shunts, mask_signals):
        trend2one = torch.abs(mask_shunts.grad[:, 0]) / mask_signals.grad[:, 0].clamp(min=1e-6)
        trend2zero = torch.abs(mask_shunts.grad[:, 1]) / mask_signals.grad[:, 1].clamp(min=1e-6)
        self.trend_accum[:, 0] += trend2one
        self.trend_accum[:, 1] += trend2zero
       
    def STIF(self, iterations, factor=0.01):             #  State-Trend Inconsistency-Filtering Assessment
        trends2one = self.trend_accum[:, 0]
        trends2zero = self.trend_accum[:, 1]
        #
        all_identity = self._identity.squeeze()
        threshold = iterations * factor
        one_mask = torch.logical_and((all_identity > 0.5), (trends2zero > threshold))
        another_mask = torch.logical_and((all_identity <= 0.5), (trends2one > threshold))
        filter_mask = torch.logical_or(one_mask, another_mask)
        print("shape of mask : ", filter_mask.shape)
        return filter_mask

    def reset_age(self):
        self.age = torch.ones((self.get_xyz.shape[0], 1), device="cuda")

    def identify_and_split(self, scene_extent, filter_mask, radii, N=2, prune_only=False, use_knn=False, split=True):
        if prune_only:
            selected_pts_mask = filter_mask
            print("Boundary points: ", selected_pts_mask.sum().item())
            self.tmp_radii = None
            self.prune_points(selected_pts_mask)
        else:
            selected_pts_mask = torch.logical_and(filter_mask,
                                    torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)   
            print("Boundary points: ", selected_pts_mask.sum().item())

            if not split:
                if use_knn:
                    self.local_knn_no_split(selected_pts_mask)
                    self.reset_age()
                    self.age[selected_pts_mask] = 0
                else:
                    print("no split, no knn, nothing.")
            else:
                if use_knn:
                    self.local_knn(selected_pts_mask, K=10)
                    new_xyz, new_identity = self.Directional_Split(selected_pts_mask, self.local_neighbor["F"], self.local_neighbor["B"], λ=0.25)
                else:
                    stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
                    means = torch.zeros((stds.size(0), 3), device="cuda")
                    samples = torch.normal(mean=means, std=stds)
                    rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                    new_identity = torch.zeros_like(self._identity[selected_pts_mask]).repeat(N, 1)
                    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
                
                new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
                new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
                new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
                new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
                new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

                self.tmp_radii = radii
                if self.tmp_radii is not None:
                    new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
                else:
                    new_tmp_radii = None
                        
                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                            new_rotation, new_identity, new_tmp_radii)
                        
                prune_filter = torch.cat(
                    (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
                self.prune_points(prune_filter)

                self.reset_age()
                new_pts_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
                new_pts_mask[-(selected_pts_mask.sum().item() * N):] = True
                self.age[new_pts_mask] = 0
                    
                tmp_radii = self.tmp_radii
                self.tmp_radii = None

    torch.cuda.empty_cache()

    def segment(self, mask=None):
        if mask is None:
            mask = self._identity > 0.5  # foreground
            # mask = self._identity < 0.5  # background
        mask = mask.squeeze()

        self._xyz = self._xyz[mask]
        self._identity = self._identity[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._opacity = self._opacity[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
    
    def training_state_switch(self, mask_state):
        if mask_state:  # mask
            self._identity.requires_grad_(True)
            self._xyz.requires_grad_(True)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._opacity.requires_grad_(True)
        else:        # rgb
            self._identity.requires_grad_(False)
            self._xyz.requires_grad_(False)
            self._scaling.requires_grad_(True)
            self._rotation.requires_grad_(True)
            self._features_dc.requires_grad_(True)
            self._features_rest.requires_grad_(True)
            self._opacity.requires_grad_(True)
    
    def local_knn_no_split(self, filter_mask, K, batch_size=1000):
        torch.cuda.reset_peak_memory_stats()
        # --- boundary Gaussians ---
        boundary_points = self.get_xyz[filter_mask]   # [N, 3]
        # --- all Gaussians ---
        all_xyz = self.get_xyz                        # [M, 3]
        global_idx = torch.arange(all_xyz.shape[0], device=all_xyz.device)
        neighbor_idx = []

        for begin in range(0, boundary_points.shape[0], batch_size):
            end = min(begin + batch_size, boundary_points.shape[0])
            batch_points = boundary_points[begin:end]

            # --- KNN ---
            dist = torch.cdist(batch_points, all_xyz)  # [B, M]
            _, knn_idx = torch.topk(
                dist,
                k=min(K, all_xyz.shape[0]),
                largest=False
            )

            neighbor_idx.append(global_idx[knn_idx])

       
        all_neighbors = torch.cat(neighbor_idx, dim=0)  # [N, K]
        self.local_neighbor = {"ALL": all_neighbors}

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[local_knn] stored {all_neighbors.shape[0]} neighbor groups (K={K})")
        print(f"[local_knn] peak GPU memory: {peak_mem:.2f} GB")
    
    @torch.no_grad()
    def local_knn(self, filter_mask, K=10, query_chunk=4196, ref_chunk=65536):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

        ambiguous_points = self.get_xyz.detach()[filter_mask]  # [N, 3]
        all_identity, all_xyz = self.get_identity.detach().squeeze(), self.get_xyz.detach()
        mask_pos, mask_neg = all_identity > 0.5, all_identity < 0.5
        xyz_pos, xyz_neg = all_xyz[mask_pos], all_xyz[mask_neg]
        global_pos, global_neg = torch.nonzero(mask_pos, as_tuple=False).squeeze(1), torch.nonzero(mask_neg, as_tuple=False).squeeze(1)

        def knn_chunked(query_points, ref_points, global_ref_idx, K):
            N, M = query_points.shape[0], ref_points.shape[0]
            if M == 0: raise RuntimeError("local_knn: reference point set is empty.")
            K_eff, all_neighbors = min(K, M), []

            for q_begin in range(0, N, query_chunk):
                q = query_points[q_begin:min(q_begin + query_chunk, N)]
                B = q.shape[0]
                best_dist = torch.full((B, K_eff), float("inf"), device=q.device, dtype=q.dtype)
                best_idx = torch.full((B, K_eff), -1, device=q.device, dtype=torch.long)

                for r_begin in range(0, M, ref_chunk):
                    r = ref_points[r_begin:min(r_begin + ref_chunk, M)]
                    dist = torch.cdist(q, r)
                    cur_dist, cur_idx = torch.topk(dist, k=min(K_eff, r.shape[0]), dim=1, largest=False)
                    cur_idx = cur_idx + r_begin
                    cand_dist, cand_idx = torch.cat([best_dist, cur_dist], dim=1), torch.cat([best_idx, cur_idx], dim=1)
                    best_dist, order = torch.topk(cand_dist, k=K_eff, dim=1, largest=False)
                    best_idx = torch.gather(cand_idx, 1, order)
                    del dist, cur_dist, cur_idx, cand_dist, cand_idx, order

                all_neighbors.append(global_ref_idx[best_idx])

            return torch.cat(all_neighbors, dim=0)

        all_neighbor_pos = knn_chunked(ambiguous_points, xyz_pos, global_pos, K)
        all_neighbor_neg = knn_chunked(ambiguous_points, xyz_neg, global_neg, K)

        self.local_neighbor = {"F": all_neighbor_pos, "B": all_neighbor_neg}
    
    def Directional_Split(self, selected_pts_mask, neighbor_pos, neighbor_neg, λ=0.5):
        device = self.get_xyz.device
        xyz_sel = self.get_xyz[selected_pts_mask]  # [N, 3]

        mean_pos = self.get_xyz[neighbor_pos].mean(dim=1)  # [N, 3]
        mean_neg = self.get_xyz[neighbor_neg].mean(dim=1)  # [N, 3]

        new_xyz_pos = xyz_sel + λ * (mean_pos - xyz_sel)
        new_xyz_neg = xyz_sel + λ * (mean_neg - xyz_sel)

        new_xyz = torch.cat([new_xyz_pos, new_xyz_neg], dim=0)  # [2N, 3]

        new_identity_pos = torch.full((xyz_sel.shape[0], 1), 0.9, device=device)
        new_identity_neg = torch.full((xyz_sel.shape[0], 1), 0.1, device=device)
        new_identity = torch.cat([new_identity_pos, new_identity_neg], dim=0)  # [2N, 1]

        return new_xyz, new_identity

    def neighbor_consistency_loss(self, w=(1.0, 1.0), split=True):
        assert hasattr(self, "local_neighbor"), \
            "call self.local_knn(...) before neighbor_consistency_loss()"

        if split:
            pos_idx, neg_idx = self.local_neighbor["F"], self.local_neighbor["B"]  # [N, K]
            N, K = pos_idx.shape

            new_idx = torch.where(self.age.squeeze() == 0)[0]
            assert new_idx.numel() >= 2 * N, \
                f"need at least {2 * N} new points, got {new_idx.numel()}"

            new_pos, new_neg = new_idx[:N], new_idx[N:2 * N]

            opacity = self.get_opacity.squeeze(-1)  # [M]
            scale = self.get_scaling              # [M, 3] or similar

            opacity_pos, opacity_neg = opacity[pos_idx], opacity[neg_idx]  # [N, K]
            scale_pos, scale_neg = scale[pos_idx], scale[neg_idx]          # [N, K, ...]

            # opacity L2
            o_pos = (opacity[new_pos].unsqueeze(1) - opacity_pos).pow(2).mean(dim=-1)
            o_neg = (opacity[new_neg].unsqueeze(1) - opacity_neg).pow(2).mean(dim=-1)

            # scale L1
            s_pos = (scale[new_pos].unsqueeze(1) - scale_pos).abs().mean(dim=(-1, -2))
            s_neg = (scale[new_neg].unsqueeze(1) - scale_neg).abs().mean(dim=(-1, -2))

            wo, ws = w
            loss_pos = wo * o_pos + ws * s_pos
            loss_neg = wo * o_neg + ws * s_neg

            return 0.5 * (loss_pos.mean() + loss_neg.mean())

        else:
            neighbor_idx = self.local_neighbor["ALL"]  # [N, K]
            N, K = neighbor_idx.shape

            # boundary Gaussians
            boundary_idx = torch.where(self.age.squeeze() == 0)[0]
            assert boundary_idx.numel() == N, \
                f"boundary_idx has {boundary_idx.numel()} elements, but neighbor_idx has N={N}"

            opacity = self.get_opacity.squeeze(-1)  # [M]
            scale = self.get_scaling                # [M, 3] or similar

            opacity_nb = opacity[neighbor_idx]      # [N, K]
            scale_nb = scale[neighbor_idx]          # [N, K, ...]

            opacity_c = opacity[boundary_idx]       # [N]
            scale_c = scale[boundary_idx]           # [N, ...]

            # opacity L2
            o_loss = (opacity_c.unsqueeze(1) - opacity_nb).pow(2).mean(dim=-1)

            # scale L1
            s_loss = (scale_c.unsqueeze(1) - scale_nb).abs().mean(dim=(-1, -2))

            wo, ws = w
            loss = wo * o_loss + ws * s_loss

            return loss.mean()
