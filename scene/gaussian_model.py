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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.density_utils import DensityDistribution
from utils.potential_utils import VisualPotential
# from utils.distance import detect_density_geometric_keypoints
from utils.mainpoint import detect_density_geometric_keypoints
import math
import open3d as o3d
import torch.nn.functional as F



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
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.camera_densities = {}
        self.visual_potential = VisualPotential()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
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
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
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
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
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

    def store_keypoints(self,curvature_file):
        # Step 1: 从模型中获取点云数据 (Tensor -> NumPy)
        pcd_array = self.get_xyz.detach().cpu().numpy()  # 将点云从 GPU 张量转为 NumPy 数组

        if pcd_array.shape[0] == 0:
            raise ValueError("点云数据为空，无法构建 KDTree，请检查输入点云。")

        # Step 2: 转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array)  # 将 NumPy 数组赋值为 Open3D 点云的点

        if len(pcd.points) == 0:
            raise ValueError("点云数据为空，无法构建 KDTree，请检查输入点云。")

        # Step 3: 检测关键点
        keypoints = detect_density_geometric_keypoints(
            pcd,
            curvature_file,
            radius=0.1,
            density_threshold=200,
            curvature_threshold=0.6,
            nms_radius=3.0,
            max_keypoints=100 
        )

        # Step 4: 将关键点坐标转换为 PyTorch 张量并存储到显卡上
        keypoints_array = np.asarray(keypoints.points)
        self.keypoints = torch.tensor(keypoints_array, dtype=torch.float32, device="cuda")

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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        device = self.get_xyz.device
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # 获取多视角综合势能（平均）
        valid_potentials = []
        for cam_id in self.visual_potential.point_potentials:
            pot = self.visual_potential.get_potential(cam_id)
            if pot is not None and len(pot) == n_init_points:
                valid_potentials.append(pot)
        avg_potential = torch.stack(valid_potentials).mean(dim=0)+self.visual_potential.get_potential(self.current_camera_id) if valid_potentials else None

        if avg_potential is not None:
            padded_potential = torch.zeros_like(padded_grad)
            padded_potential[:avg_potential.shape[0]] = avg_potential
            split_score = padded_grad * (1.0 + padded_potential) * 1.2
        else:
            split_score = padded_grad

        selected_pts_mask = torch.where(split_score >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                        torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        if selected_pts_mask.sum() == 0:
            self.densify_and_generate(scene_extent)
            return
        #parent_indices = torch.nonzero(selected_pts_mask).squeeze(1)
        # 计算每个选中点的分裂数量
        K = selected_pts_mask.sum()
        if avg_potential is not None:
            selected_potentials = padded_potential[selected_pts_mask]
            base = torch.ones(K, dtype=torch.long, device=device)
            remaining = K * (N - 1)  # 剩余需要分配的数目

            if remaining > 0:
                # 优化1：使用势能的平方作为权重，放大高势能点的影响
                weights = selected_potentials ** 2  
                sum_weights = weights.sum()

                if sum_weights <= 1e-6:  # 防止除零
                    # 优化2：当权重无效时，回退到ceil直接计算
                    N_adjusted = torch.ceil(N * (1.0 + selected_potentials)).long()
                else:
                    # 动态分配剩余次数（更积极版本）
                    adjusted_remaining_float = (remaining * weights) / sum_weights
                    adjusted_remaining = torch.ceil(adjusted_remaining_float).long()  # 优化3：使用ceil而非floor
                    total_adjusted = adjusted_remaining.sum()
                    
                    # 处理可能的溢出
                    if total_adjusted > remaining:
                        adjusted_remaining = (adjusted_remaining * remaining / total_adjusted).long()
                    elif total_adjusted < remaining:
                        # 将剩余次数分配给权重最大的点
                        remainder = remaining - total_adjusted
                        top_indices = torch.topk(weights, remainder, largest=True).indices
                        adjusted_remaining[top_indices] += 1
                    
                    base += adjusted_remaining
                
                # 优化4：确保每个点至少分裂N次
                N_adjusted = torch.clamp_min(base, N)
            else:
                N_adjusted = base
        else:
            N_adjusted = torch.full((K,), N, device=device)

        # 计算总的新点数量
        total_new_points = N_adjusted.sum().item()
        
        # 创建重复索引
        selected_pts = torch.nonzero(selected_pts_mask).squeeze()
        indices = torch.repeat_interleave(torch.arange(len(selected_pts), device=device), N_adjusted)

        # 获取选中点的缩放值并重复
        stds = self.get_scaling[selected_pts_mask][indices]  # shape: [total_new_points, 3]
        
        # 生成随机偏移
        means = torch.zeros((total_new_points, 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        
        # 应用旋转
        rots = build_rotation(self._rotation[selected_pts_mask][indices])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                self.get_xyz[selected_pts_mask][indices]

        # 计算新的缩放值 - 修复维度问题
        scaling_factor = torch.repeat_interleave(N_adjusted, N_adjusted).unsqueeze(-1).to(device)  # [total_new_points, 1]
        new_scaling = self.scaling_inverse_activation(stds / (0.8 * scaling_factor))

        # 其他属性
        new_rotation = self._rotation[selected_pts_mask][indices]
        new_features_dc = self._features_dc[selected_pts_mask][indices]
        new_features_rest = self._features_rest[selected_pts_mask][indices]
        new_opacity = self._opacity[selected_pts_mask][indices]
        new_tmp_radii = self.tmp_radii[selected_pts_mask][indices]

        # 更新点云
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                            new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 创建剪枝掩码
        prune_filter = torch.cat((selected_pts_mask,
                            torch.zeros(total_new_points, device=device, dtype=bool)))
        self.prune_points(prune_filter)
        current_num_points = self.get_xyz.shape[0]

        # 遍历所有缓存的相机势能
        for cam_id in list(self.visual_potential.point_potentials.keys()):
            potential_info = self.visual_potential.point_potentials[cam_id]
            existing_potential = potential_info['potential']
            
            if existing_potential.shape[0] < current_num_points:
                num_new_points = current_num_points - existing_potential.shape[0]
                
                # 获取父点势能并衰减
                parent_potentials = existing_potential[selected_pts]
                repeated_potentials = torch.repeat_interleave(parent_potentials, N_adjusted)
                new_potential = repeated_potentials[:num_new_points] * 0.8  # 衰减系数0.8
                
                updated_potential = torch.cat([existing_potential, new_potential], dim=0)
                self.visual_potential.point_potentials[cam_id]['potential'] = updated_potential


    # def densify_and_clone_1(self, grads, grad_threshold, scene_extent, scenelist):
    #     """基于梯度与视觉势能进行点克隆，优先在视觉重要区域和边界区域增加点密度"""
    #     # 获取当前相机的视觉势能
    #     valid_potentials = []
    #     n_init_points = self.get_xyz.shape[0]
    #     for cam_id in self.visual_potential.point_potentials:
    #         pot = self.visual_potential.get_potential(cam_id)
    #         if pot is not None and len(pot) == n_init_points:
    #             valid_potentials.append(pot)
        
    #     # 计算平均势能和当前相机势能的加权和，加大当前相机权重
    #     current_pot = self.visual_potential.get_potential(self.current_camera_id)
    #     avg_potential = torch.stack(valid_potentials).mean(dim=0) * 0.4 + current_pot * 0.6 if valid_potentials else current_pot
        
    #     if avg_potential is not None and len(avg_potential) == self.get_xyz.shape[0]:
    #         # 归一化势能到[-1,1]区间
    #         potential_normalized = 2 * (avg_potential - avg_potential.min()) / (avg_potential.max() - avg_potential.min() + 1e-8) - 1
            
    #         # 使用双曲正切函数非线性增强高势能区域
    #         potential_enhanced = torch.tanh(2.5 * potential_normalized)
            
    #         # 使用梯度幅度作为边界检测的简化方法
    #         # 梯度大的地方往往是边界区域
    #         grad_norms = torch.norm(grads, dim=-1)
    #         # 归一化梯度幅度作为边界指示器
    #         normalized_grads = grad_norms / (torch.max(grad_norms) + 1e-8)
            
    #         # 边界增强因子：将梯度幅度作为边界指示器
    #         boundary_factor = torch.pow(normalized_grads, 0.5)  # 平方根使分布更均匀
            
    #         # 综合考虑势能和边界因素（高势能或高梯度区域都获得优先）
    #         combined_enhancement = torch.clamp(potential_enhanced + 0.3 * boundary_factor, min=-1.0, max=1.0)
            
    #         # 动态调整梯度阈值：势能越高或在边界附近阈值越低
    #         effective_grad_threshold = grad_threshold * (1 - 0.7 * combined_enhancement)
            
    #         # 基于梯度和动态阈值选择点
    #         selected_pts_mask = (grad_norms >= effective_grad_threshold)
            
    #         # 增加对低势能区域的限制
    #         low_potential_mask = potential_enhanced < -0.5
    #         if torch.any(low_potential_mask):
    #             # 只在低势能区域随机保留部分点，防止过度分裂
    #             low_potential_probs = torch.rand_like(grad_norms[low_potential_mask])
    #             # 根据梯度大小调整概率（梯度越大，保留概率越高）
    #             keep_threshold = 0.7 - 0.5 * normalized_grads[low_potential_mask]  # 0.2-0.7的动态阈值
    #             selected_pts_mask[low_potential_mask] = low_potential_probs < keep_threshold
            
    #         # 动态调整缩放条件
    #         max_scaling = torch.max(self.get_scaling, dim=1).values
    #         # 只考虑正值势能影响
    #         scaling_enhancement = torch.clamp(combined_enhancement, min=0)
    #         # 动态缩放阈值
    #         dynamic_scaling_threshold = self.percent_dense * scene_extent * (1 + 0.8 * scaling_enhancement)
    #         scaling_condition = max_scaling <= dynamic_scaling_threshold
            
    #         # 为高梯度点（可能是边界）放宽缩放限制
    #         high_grad_mask = normalized_grads > 0.7  # 高梯度点
    #         if torch.any(high_grad_mask):
    #             # 高梯度点的缩放阈值提高40%
    #             scaling_condition[high_grad_mask] = max_scaling[high_grad_mask] <= 1.4 * self.percent_dense * scene_extent
    #     else:
    #         # 无势能信息时回退到原始逻辑
    #         selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
    #         scaling_condition = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        
    #     # 结合梯度条件和缩放条件
    #     selected_pts_mask = torch.logical_and(selected_pts_mask, scaling_condition)
        
    #     # 克隆选中的点
    #     new_xyz = self._xyz[selected_pts_mask]
    #     new_features_dc = self._features_dc[selected_pts_mask]
    #     new_features_rest = self._features_rest[selected_pts_mask]
    #     new_opacities = self._opacity[selected_pts_mask]
    #     new_scaling = self._scaling[selected_pts_mask]
    #     new_rotation = self._rotation[selected_pts_mask]
    #     new_tmp_radii = self.tmp_radii[selected_pts_mask]
        
    #     # 只为边界点添加微小随机偏移，促进边界扩展
    #     if avg_potential is not None:
    #         # 简化版：使用高梯度作为边界指示器
    #         boundary_mask = (normalized_grads > 0.7) & selected_pts_mask
    #         boundary_indices = torch.nonzero(boundary_mask).squeeze()
            
    #         if boundary_indices.numel() > 0:
    #             # 处理单个索引的情况
    #             if boundary_indices.dim() == 0:
    #                 boundary_indices = boundary_indices.unsqueeze(0)
                
    #             # 边界点索引映射
    #             boundary_map = torch.zeros(selected_pts_mask.sum(), dtype=torch.bool, device=self._xyz.device)
    #             for idx in boundary_indices:
    #                 idx_in_selected = (selected_pts_mask.cumsum(0)[idx] - 1).item()
    #                 if 0 <= idx_in_selected < boundary_map.shape[0]:
    #                     boundary_map[idx_in_selected] = True
                
    #             # 为边界点添加随机偏移，扩展边界
    #             if boundary_map.any():
    #                 # 计算平均点半径作为偏移基准
    #                 avg_radius = torch.max(new_scaling[boundary_map], dim=1).values.mean() * 0.1
    #                 # 生成随机方向
    #                 random_dirs = torch.randn(boundary_map.sum(), 3, device=new_xyz.device)
    #                 random_dirs = random_dirs / (torch.norm(random_dirs, dim=1, keepdim=True) + 1e-8)
    #                 # 应用随机偏移
    #                 new_xyz[boundary_map] = new_xyz[boundary_map] + random_dirs * avg_radius.item()

    #     # 应用标准的密度后续处理
    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
    #                             new_opacities, new_scaling, new_rotation, new_tmp_radii)

    #     # 更新所有缓存的势能数组
    #     current_num_points = self.get_xyz.shape[0]
    #     for cam_id in list(self.visual_potential.point_potentials.keys()):
    #         potential_info = self.visual_potential.point_potentials[cam_id]
    #         existing_potential = potential_info['potential']
            
    #         if existing_potential.shape[0] < current_num_points:
    #             num_new_points = current_num_points - existing_potential.shape[0]
                
    #             # 获取母点势能
    #             parent_indices = torch.nonzero(selected_pts_mask).squeeze()
    #             parent_potentials = existing_potential[parent_indices]
                
    #             # 对于边界点，保持较高的势能传递比例
    #             if avg_potential is not None:
    #                 # 使用梯度作为边界指示器
    #                 is_boundary = normalized_grads[parent_indices] > 0.7
    #                 transfer_ratio = torch.ones_like(parent_potentials) * 0.8  # 基础传递率
    #                 transfer_ratio[is_boundary[:len(transfer_ratio)]] = 0.9  # 边界点提高到90%
    #                 new_potential = parent_potentials[:num_new_points] * transfer_ratio[:num_new_points]
    #             else:
    #                 # 原有逻辑
    #                 new_potential = parent_potentials[:num_new_points] * 0.8
                
    #             updated_potential = torch.cat([existing_potential, new_potential], dim=0)
    #         self.visual_potential.point_potentials[cam_id]['potential'] = updated_potential
    def densify_and_clone(self, grads, grad_threshold, scene_extent,scenelist):
            """基于梯度与视觉势能进行点克隆，优先在视觉重要区域增加点密度"""
            # 获取当前相机的视觉势能
            valid_potentials = []
            n_init_points = self.get_xyz.shape[0]
            for cam_id in self.visual_potential.point_potentials:
                pot = self.visual_potential.get_potential(cam_id)
                if pot is not None and len(pot) == n_init_points:
                    valid_potentials.append(pot)
            avg_potential = torch.stack(valid_potentials).mean(dim=0) +self.visual_potential.get_potential(self.current_camera_id) if valid_potentials else None
            if avg_potential is not None and len(avg_potential) == self.get_xyz.shape[0]:
                # 归一化势能到[-1,1]区间以增强对比度
                potential_normalized = 2 * (avg_potential - avg_potential.min()) / (avg_potential.max() - avg_potential.min() + 1e-8) - 1
                # 使用双曲正切函数非线性增强高势能区域
                potential_enhanced = torch.tanh(3 * potential_normalized)
                
                # 动态调整梯度阈值：势能越高阈值越低

                effective_grad_threshold = grad_threshold * (1 - 0.8 * potential_enhanced)
                # 计算调整后的梯度范数
                grad_norms = torch.norm(grads, dim=-1)
                selected_pts_mask = (grad_norms >= effective_grad_threshold)

                # 动态调整缩放条件：势能越高允许的缩放越大
                max_scaling = torch.max(self.get_scaling, dim=1).values
                # 势能增强因子（只考虑正值影响）
                scaling_enhancement = torch.clamp(potential_enhanced, min=0)  # 负势能区域不降低要求
                # 动态计算缩放阈值（基础阈值 + 势能增强部分）
                dynamic_scaling_threshold = self.percent_dense * scene_extent * (1 + 0.8 * scaling_enhancement)
                scaling_condition = max_scaling <= dynamic_scaling_threshold
            else:
                # 无势能信息时回退到原始逻辑
                selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
                scaling_condition = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
            
            # 结合梯度条件和缩放条件
            selected_pts_mask = torch.logical_and(selected_pts_mask, scaling_condition)
            
            # 克隆选中的点
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_tmp_radii = self.tmp_radii[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                    new_opacities, new_scaling, new_rotation, new_tmp_radii)

            # 更新所有缓存的势能数组（保持原有逻辑不变）
            current_num_points = self.get_xyz.shape[0]
            for cam_id in list(self.visual_potential.point_potentials.keys()):
                potential_info = self.visual_potential.point_potentials[cam_id]
                existing_potential = potential_info['potential']
                
                if existing_potential.shape[0] < current_num_points:
                    num_new_points = current_num_points - existing_potential.shape[0]
                    
                    # 获取母点势能并衰减
                    parent_indices = torch.nonzero(selected_pts_mask).squeeze()
                    parent_potentials = existing_potential[parent_indices]
                    new_potential = parent_potentials[:num_new_points] * 0.8  # 衰减系数0.8
                    
                    updated_potential = torch.cat([existing_potential, new_potential], dim=0)
                    self.visual_potential.point_potentials[cam_id]['potential'] = updated_potential
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,scenelist,scale,itr):
        """
        对点云进行密集化处理，同时根据梯度、屏幕大小和不透明度对点进行裁剪。
        """
        # 1. 计算点梯度
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0  # 清除 NaN 值
        

        # 2. 暂存半径信息
        self.tmp_radii = radii

        # 3. 密集化点云
        # if(itr%300==0):
        #     self.densify_and_clone_1(grads, max_grad, extent,scenelist)
        # else:
        #     self.densify_and_clone(grads, max_grad, extent,scenelist)
        self.densify_and_clone(grads, max_grad, extent,scenelist)
        if (scenelist==1):
            self.densify_and_split(grads, max_grad, extent)
        if (scenelist==2):
            self.densify_and_split_in(grads, max_grad, extent)

        # 4. 计算加权质心（利用已有函数和 GPU 数据）
        # 通过距离和密度权重计算质心
        #centroid = (self.get_xyz * self.get_opacity).sum(dim=0, keepdim=True) / self.get_opacity.sum(dim=0, keepdim=True)

        # 5. 计算每个点到质心的距离
        # 获取前5个最近距离
        batch_size = 10000  # 根据你的 GPU 内存大小调整
        distances = []
        for start_idx in range(0, self.get_xyz.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, self.get_xyz.shape[0])
            distances_batch, _ = torch.cdist(self.get_xyz[start_idx:end_idx], self.keypoints).topk(3, largest=False)
            distances.append(distances_batch)
            torch.cuda.empty_cache()
        distances_to_keypoints = torch.cat(distances, dim=0)


        # 定义固定权重
        weights = torch.tensor([0.4,0.3,0.3], device=distances_to_keypoints.device)

        # 计算加权平均
        weighted_distances = (distances_to_keypoints * weights).sum(dim=1)

        # 6. 距离归一化
        max_distance = weighted_distances.max()  # 最大距离
        if max_distance == 0:
            normalized_distances = torch.zeros_like(distances)
        else:
            normalized_distances = weighted_distances / max_distance
        normalized_distances = normalized_distances   # 偏移

        sc=len(self.get_xyz)
        sc = ((sc - 1000000) // 1000000 + 1) * 2.2*0.1
        if(scenelist==1):
            if(sc<=0.5):
                sc=0.3
        if(scenelist==2):
            if(sc<=1.0):
                sc=0.1
               
            
        tanh =torch.nn.Tanh()
        normalized_distances = sc*(tanh(2*normalized_distances)-0.96)
        #normalized_distances = scale*(tanh(2*normalized_distances)-0.96)
        normalized_distances = normalized_distances.unsqueeze(1)  # 形状调整

        # 7. 增强不透明度
        opacity_strengthen = self.get_opacity + 0.050 * normalized_distances

        # 8. 根据条件进行裁剪
        prune_mask = (opacity_strengthen < min_opacity).squeeze()  # 不透明度低于阈值的点

        # 9. 如果有屏幕大小限制，根据屏幕大小和比例进一步裁剪
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  # 屏幕尺寸过大的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 世界坐标尺度过大的点
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 10. 对点云进行裁剪
        self.prune_points(prune_mask)

        # 11. 清理缓存
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()
    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,scenelist,scale,itr):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     self.tmp_radii = radii
    #     self.densify_and_clone(grads, max_grad, extent,scenelist)
    #     if (scenelist==1):
    #         self.densify_and_split(grads, max_grad, extent)
    #     if (scenelist==2):
    #         self.densify_and_split_in(grads, max_grad, extent)

    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)
    #     tmp_radii = self.tmp_radii
    #     self.tmp_radii = None

    #     torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 计算二维梯度范数
        grads = viewspace_point_tensor.grad[update_filter, :2]
        grad_norms = torch.norm(grads, dim=-1, keepdim=True)
        
        # 原始梯度累积
        self.xyz_gradient_accum[update_filter] += grad_norms
        
        # 统计更新次数
        self.denom[update_filter] += 1

    def world_to_3d(self,points, transform_matrix):
        """
        将世界坐标点通过变换矩阵转换为另一个三维空间（如相机坐标系）中的坐标。

        Args:
            points: Tensor of shape (N, 3) 世界坐标系中的点。
            transform_matrix: 4x4 变换矩阵（例如世界到相机的视图矩阵）。
        
        Returns:
            transformed_coords: Tensor of shape (N, 3) 目标三维空间中的坐标。
        """
        # 转换为齐次坐标 (N, 4)
        homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        
        # 应用变换矩阵（例如世界到相机）
        transformed_homo = homogeneous @ transform_matrix
        
        # 透视除法（确保处理可能的投影情况）
        transformed_coords = transformed_homo[:, :3] / transformed_homo[:, 3:4]
        
        return transformed_coords
    
    def densify_and_split_in(self, grads, grad_threshold, scene_extent, N=2):           
            device = self.get_xyz.device
            n_init_points = self.get_xyz.shape[0]
            padded_grad = torch.zeros((n_init_points), device=device)
            padded_grad[:grads.shape[0]] = grads.squeeze()

            # 获取多视角综合势能（平均）
            valid_potentials = []
            for cam_id in self.visual_potential.point_potentials:
                pot = self.visual_potential.get_potential(cam_id)
                if pot is not None and len(pot) == n_init_points:
                    valid_potentials.append(pot)
            avg_potential = torch.stack(valid_potentials).mean(dim=0)+self.visual_potential.get_potential(self.current_camera_id) if valid_potentials else None

            if avg_potential is not None:
                padded_potential = torch.zeros_like(padded_grad)
                padded_potential[:avg_potential.shape[0]] = avg_potential
                split_score = padded_grad * (1.0 + padded_potential) * 1.2
            else:
                split_score = padded_grad
            # 选择需要分裂的点
            selected_pts_mask = torch.where(split_score >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

            if selected_pts_mask.sum() == 0:
                return
            parent_indices = torch.nonzero(selected_pts_mask).squeeze(1)
            # 计算每个选中点的分裂数量
            K = selected_pts_mask.sum()
            if avg_potential is not None:
                selected_potentials = padded_potential[selected_pts_mask]
                base = torch.ones(K, dtype=torch.long, device=device)
                remaining = K * (N - 1)  # 剩余需要分配的数目

                if remaining > 0:
                    weights = selected_potentials  # 使用势能作为权重
                    sum_weights = weights.sum()

                    if sum_weights <= 0:
                        # 所有势能为0或负，平均分配剩余次数
                        per_point = torch.div(remaining, K, rounding_mode='trunc')
                        remainder = remaining % K
                        additional = torch.full((K,), per_point, device=device, dtype=torch.long)
                        additional[:remainder] += 1
                        base += additional
                    else:
                        # 计算每个点的剩余次数
                        adjusted_remaining_float = (remaining * weights) / sum_weights
                        adjusted_remaining_base = torch.floor(adjusted_remaining_float).long()
                        remainders = adjusted_remaining_float - adjusted_remaining_base

                        # 计算总分配次数和剩余次数
                        total_adjusted = adjusted_remaining_base.sum().item()
                        R = remaining - total_adjusted

                        if R > 0:
                            # 找到余数最大的R个点
                            _, indices = torch.topk(remainders, R)
                            adjusted_remaining_base[indices] += 1

                        base += adjusted_remaining_base
                N_adjusted = base
            else:
                N_adjusted = torch.full((K,), N, device=device)

            # 计算总的新点数量
            total_new_points = N_adjusted.sum().item()
            
            # 创建重复索引
            selected_pts = torch.nonzero(selected_pts_mask).squeeze()
            indices = torch.repeat_interleave(torch.arange(len(selected_pts), device=device), N_adjusted)

            # 获取选中点的缩放值并重复
            stds = self.get_scaling[selected_pts_mask][indices]  # shape: [total_new_points, 3]
            
            # 生成随机偏移
            means = torch.zeros((total_new_points, 3), device=device)
            samples = torch.normal(mean=means, std=stds)
            
            # 应用旋转
            rots = build_rotation(self._rotation[selected_pts_mask][indices])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                    self.get_xyz[selected_pts_mask][indices]

            # 计算新的缩放值 - 修复维度问题
            scaling_factor = torch.repeat_interleave(N_adjusted, N_adjusted).unsqueeze(-1).to(device)  # [total_new_points, 1]
            new_scaling = self.scaling_inverse_activation(stds / (0.8 * scaling_factor))

            # 其他属性
            new_rotation = self._rotation[selected_pts_mask][indices]
            new_features_dc = self._features_dc[selected_pts_mask][indices]
            new_features_rest = self._features_rest[selected_pts_mask][indices]
            new_opacity = self._opacity[selected_pts_mask][indices]
            new_tmp_radii = self.tmp_radii[selected_pts_mask][indices]

            # 更新点云
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                new_opacity, new_scaling, new_rotation, new_tmp_radii)

            # 创建剪枝掩码
            prune_filter = torch.cat((selected_pts_mask,
                                torch.zeros(total_new_points, device=device, dtype=bool)))
            self.prune_points(prune_filter)
            current_num_points = self.get_xyz.shape[0]

            # 遍历所有缓存的相机势能
            for cam_id in list(self.visual_potential.point_potentials.keys()):
                potential_info = self.visual_potential.point_potentials[cam_id]
                existing_potential = potential_info['potential']
                
                # 检查当前势能长度是否匹配点数
                if existing_potential.shape[0] < current_num_points:
                    # 计算需要新增的点数
                    num_new_points = current_num_points - existing_potential.shape[0]
                    
                    # 初始化新点势能为父点势能的均值或零
                    if num_new_points > 0:
                        new_potential = torch.zeros(num_new_points, 
                                                device=existing_potential.device,
                                                dtype=existing_potential.dtype)
                        
                        # 若希望继承父点部分势能（可选）
                        # selected_parents = ...  # 确定新点的父索引
                        # new_potential = existing_potential[selected_parents] * 0.5
                        
                        # 拼接并更新势能
                        updated_potential = torch.cat([existing_potential, new_potential], dim=0)
                        self.visual_potential.point_potentials[cam_id]['potential'] = updated_potential

    def densify_and_generate(self, scene_extent, num_candidates=500, density_threshold=0.1, max_new_points=1000):
        """主动在低密度区域生成新点，结合势能与密度信息"""
        device = self.get_xyz.device
        if self.get_xyz.shape[0] == 0:
            return

        # 动态计算场景边界
        scene_min, _ = torch.min(self.get_xyz, dim=0)
        scene_max, _ = torch.max(self.get_xyz, dim=0)
        scene_size = scene_max - scene_min
        
        # 扩展边界防止过小
        min_extent = 0.3 * scene_extent
        scene_min = torch.where(scene_size < min_extent, scene_min - min_extent / 2, scene_min)
        scene_max = torch.where(scene_size < min_extent, scene_max + min_extent / 2, scene_max)
        
        # 生成候选点
        candidate_xyz = torch.rand((num_candidates, 3), device=device) * (scene_max - scene_min) + scene_min
        
        # 计算最近邻距离
        with torch.no_grad():
            dists = torch.cdist(candidate_xyz, self.get_xyz).min(dim=1)[0]
        
        # 筛选低密度区域
        threshold = density_threshold * scene_extent
        selected_mask = dists > threshold
        selected_candidates = candidate_xyz[selected_mask]
        
        if selected_candidates.shape[0] == 0:
            return

        # 计算这些候选点的势能（越大表示越需要生成点）
        potential = self.visual_potential.get_potential(self.current_camera_id)
        if potential is not None:
            dist_to_potential = torch.cdist(selected_candidates, self.get_xyz).min(dim=1)[0]
            weighted_potential = potential[:selected_candidates.shape[0]] * (1 / (dist_to_potential + 1e-6))  # 通过距离加权势能
            selected_candidates = selected_candidates[weighted_potential > 0.5]  # 筛选较强势能的点
        else:
            selected_candidates = selected_candidates  # 如果没有势能信息，直接使用选中的候选点

        # 限制新生成点的数量
        new_points_count = min(selected_candidates.shape[0], max_new_points)
        
        # 初始化新点的属性
        num_new = new_points_count
        new_scaling = self.scaling_inverse_activation(torch.ones((num_new, 3), device=device) * 0.01)
        new_rotation = torch.randn((num_new, 4), device=device)
        new_rotation = F.normalize(new_rotation, dim=-1)
        new_features_dc = torch.zeros_like(self._features_dc[:1].expand(num_new, -1, -1))
        new_features_rest = torch.zeros_like(self._features_rest[:1].expand(num_new, -1, -1))
        new_opacity = self.opacity_activation(torch.ones((num_new, 1), device=device) * 0.1)
        new_tmp_radii = torch.zeros(num_new, device=device)
        
        # 继承附近点的属性：选择距离新点最近的点，作为新点的初始化参数
        for i in range(num_new):
            nearest_idx = torch.argmin(torch.cdist(selected_candidates[i:i+1], self.get_xyz), dim=1)
            new_scaling[i] = self.get_scaling[nearest_idx]
            new_rotation[i] = self.get_rotation[nearest_idx]
            new_features_dc[i] = self.get_features_dc[nearest_idx]
            new_features_rest[i] = self.get_features_rest[nearest_idx]
            new_opacity[i] = self.get_opacity[nearest_idx]

        # 更新点云
        self.densification_postfix(selected_candidates[:num_new], new_features_dc, new_features_rest,
                                    new_opacity, new_scaling, new_rotation, new_tmp_radii)
        
        # 初始化新点势能
        current_num_points = self.get_xyz.shape[0]
        for cam_id in list(self.visual_potential.point_potentials.keys()):
            potential_info = self.visual_potential.point_potentials[cam_id]
            existing_potential = potential_info['potential']
            if existing_potential.shape[0] < current_num_points:
                num_new_points = current_num_points - existing_potential.shape[0]
                new_potential = torch.ones(num_new_points, device=device) * 0.8  # 高初始势能
                updated_potential = torch.cat([existing_potential, new_potential], dim=0)
                self.visual_potential.point_potentials[cam_id]['potential'] = updated_potential








