import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import torchvision.models as models
import time

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).cuda()
        blocks = []
        blocks.extend([vgg.features[:4].eval()])
        blocks.extend([vgg.features[4:9].eval()])
        blocks.extend([vgg.features[9:16].eval()])
        self.layer_weights = [0.2, 0.3, 1.0]
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

        # Move mean and std tensors to CUDA during initialization
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))
        del vgg

    def forward(self, input, target, normalize=True):
        # Ensure input and target are on the same device as the model
        input = input.to(self.mean.device)
        target = target.to(self.mean.device)

        if normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss_map = torch.zeros_like(input[:, :1, :, :])  # [B,1,H,W]
        x=input
        y=target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            layer_loss = torch.abs(x - y).mean(dim=1, keepdim=True)
            if self.resize:
                layer_loss = F.interpolate(layer_loss, (224,224), mode='bilinear')
            else:
                layer_loss = F.interpolate(layer_loss, input.shape[-2:], mode='bilinear')
            loss_map += layer_loss * self.layer_weights[i]
        return loss_map.squeeze(1)
class VisualPotential:
    def __init__(self, block_size=(16, 16), cache_size=100, momentum=0.9):
        """
        初始化视觉势能计算器
        
        Args:
            block_size: 图像分块大小
            cache_size: 缓存的最大视角数量
            momentum: 势能更新的动量因子
        """
        self.block_size = block_size
        self.cache_size = cache_size
        self.momentum = momentum
        
        # 存储每个视角的点势能
        self.point_potentials = defaultdict(dict)
        # 存储每个视角的重要性图
        self.importance_maps = {}

        self.perceptual_loss = VGGPerceptualLoss(resize=True)
        
        # 预计算 Sobel 算子
        self.register_sobel_kernels()
    
    @staticmethod
    def haar_wavelet_decomposition(x):
        """
        执行单层Haar小波变换，支持多通道输入
        Args:
            x: 输入张量 [B, C, H, W] 或 [C, H, W]
        Returns:
            四个子带: LL, LH, HL, HH，保持输入的通道数
        """
        # 确保输入是4D张量 [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        batch_size, channels, height, width = x.shape
        
        # 确保Haar滤波器与输入数据类型匹配
        haar_h = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], 
                            device=x.device, 
                            dtype=x.dtype).reshape(1, 1, 1, 2)
        
        haar_l = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], 
                            device=x.device, 
                            dtype=x.dtype).reshape(1, 1, 1, 2)
        
        # 确保输入尺寸是偶数
        if width % 2 != 0 or height % 2 != 0:
            pad_right = 1 if width % 2 != 0 else 0
            pad_bottom = 1 if height % 2 != 0 else 0
            x = F.pad(x, (0, pad_right, 0, pad_bottom))
        
        # 对每个通道分别进行小波变换
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        
        for c in range(channels):
            x_c = x[:, c:c+1]  # 处理单个通道 [B, 1, H, W]
            
            # 水平方向滤波
            x_l = F.conv2d(x_c.transpose(-2,-1), haar_l, stride=2, padding=0)
            x_h = F.conv2d(x_c.transpose(-2,-1), haar_h, stride=2, padding=0)
            x_l = x_l.transpose(-2,-1)
            x_h = x_h.transpose(-2,-1)
            
            # 垂直方向滤波
            ll = F.conv2d(x_l, haar_l.transpose(-2,-1), stride=2, padding=0)
            lh = F.conv2d(x_l, haar_h.transpose(-2,-1), stride=2, padding=0)
            hl = F.conv2d(x_h, haar_l.transpose(-2,-1), stride=2, padding=0)
            hh = F.conv2d(x_h, haar_h.transpose(-2,-1), stride=2, padding=0)
            
            ll_list.append(ll)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)
        
        # 合并所有通道
        ll = torch.cat(ll_list, dim=1)
        lh = torch.cat(lh_list, dim=1)
        hl = torch.cat(hl_list, dim=1)
        hh = torch.cat(hh_list, dim=1)
        
        return ll, lh, hl, hh

    @staticmethod
    def wavelet_reconstruction_error(rendered_image, gt_image, level=1):
        """
        使用Haar小波变换计算重建误差
        Args:
            rendered_image: 重建图像 [B, C, H, W] 或 [C, H, W]
            gt_image: 目标图像 [B, C, H, W] 或 [C, H, W]
            level: 小波分解层数
        Returns:
            基于小波系数的重建误差（标量）
        """
        error = 0
        current_rendered = rendered_image
        current_gt = gt_image
        
        for _ in range(level):
            # 对两个图像进行小波分解
            ll_r, lh_r, hl_r, hh_r = VisualPotential.haar_wavelet_decomposition(current_rendered)
            ll_g, lh_g, hl_g, hh_g = VisualPotential.haar_wavelet_decomposition(current_gt)
            
            # 计算各个子带的误差（考虑所有通道）
            error += torch.abs(ll_r - ll_g).mean() * 5  # 低频分量误差
            error += torch.abs(lh_r - lh_g).mean() * 2  # 水平细节误差
            error += torch.abs(hl_r - hl_g).mean() * 2  # 垂直细节误差
            error += torch.abs(hh_r - hh_g).mean() * 2  # 对角细节误差
            
            # 更新当前图像为低频分量，用于下一层分解
            current_rendered = ll_r
            current_gt = ll_g
        
        return error / level
    @staticmethod
    def wavelet_reconstruction_error_map(rendered, gt, level=3):
        # Ensure input tensors have a batch dimension
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
            gt = gt.unsqueeze(0)
        
        error_map = torch.zeros_like(rendered)
        current_r, current_gt = rendered, gt

        for _ in range(level):
            ll_r, lh_r, hl_r, hh_r = VisualPotential.haar_wavelet_decomposition(current_r)
            ll_gt, lh_gt, hl_gt, hh_gt = VisualPotential.haar_wavelet_decomposition(current_gt)

            for sub_r, sub_gt, weight in zip(
                [ll_r, lh_r, hl_r, hh_r],
                [ll_gt, lh_gt, hl_gt, hh_gt],
                [5.0, 2.0, 2.0, 2.0]
            ):
                err = torch.abs(sub_r - sub_gt).mean(dim=1, keepdim=True)
                err = F.interpolate(err, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
                error_map += err * weight  # Now works due to batch dimension

            current_r, current_gt = ll_r, ll_gt

        # Squeeze batch dimension if necessary
        return error_map.squeeze(1) if error_map.dim() > 3 else error_map


    def register_sobel_kernels(self):
        """预计算 Sobel 算子用于边缘检测"""
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]], device="cuda").float().unsqueeze(0).unsqueeze(0)
        
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], device="cuda").float().unsqueeze(0).unsqueeze(0)

    def compute_view_importance(self, rendered_image, gt_image, full_proj_transform, camera_id,scenelist):
        """
        计算视图的重要性图
        
        Args:
            rendered_image: 渲染的图像
            gt_image: 真实图像
            full_proj_transform: 4x4 投影矩阵
            camera_id: 相机ID
            
        Returns:
            importance_map: 重要性图，值范围[0,1]
        """
        # 1. 计算重建误差
        if scenelist==1:
            reconstruction_error = torch.abs(rendered_image - gt_image).mean(dim=0) + \
                                self.wavelet_reconstruction_error(rendered_image, gt_image, level=3)*2.0
        else:
            reconstruction_error = torch.abs(rendered_image - gt_image).mean(dim=0) + \
                                self.wavelet_reconstruction_error(rendered_image, gt_image, level=3)*2.0
        # if(scenelist==2):
        #     reconstruction_error = -torch.abs(rendered_image - gt_image).mean(dim=0) + \
        #                             self.wavelet_reconstruction_error(rendered_image, gt_image, level=3) * 0.5 +\
        #                             self.perceptual_loss(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))

        
        # 2. 计算图像梯度
        if gt_image.dim() == 3:
            gt_gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            gt_gray = gt_image.mean(dim=1, keepdim=True)
            
        grad_x = F.conv2d(gt_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gt_gray, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))[0, 0]
        wavelet_error = self.wavelet_reconstruction_error_map(rendered_image, gt_image).squeeze(0)
        wavelet_error=wavelet_error.mean(dim=0)
        wavelet_size = wavelet_error.shape[-2:]

        perceptual_loss = self.perceptual_loss(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
        perceptual_loss_resized = F.interpolate(perceptual_loss.unsqueeze(0), size=wavelet_size, mode='bilinear', align_corners=False).squeeze(0)
        perceptual_loss_resized=perceptual_loss_resized.mean(dim=0)

        origin_loss=torch.abs(rendered_image - gt_image).squeeze(0)
        origin_loss=origin_loss.mean(dim=0)


        # 3. 结合重建误差和梯度
        importance = reconstruction_error * (1+2*gradient_magnitude+wavelet_error+perceptual_loss_resized-origin_loss )
        
        # 4. 分块并计算局部统计
        h, w = importance.shape
        block_h, block_w = self.block_size
        
        padded_h = ((h - 1) // block_h + 1) * block_h
        padded_w = ((w - 1) // block_w + 1) * block_w
        
        # 填充到块大小的整数倍
        importance_padded = F.pad(
            importance,
            (0, padded_w - w, 0, padded_h - h),
            mode='constant',
            value=0
        )
        
        # 重塑为块
        blocks = importance_padded.unfold(0, block_h, block_h).unfold(1, block_w, block_w)
        
        # 计算每个块的统计量
        block_means = blocks.mean(dim=(2, 3))
        block_vars = blocks.var(dim=(2, 3))
        
        # 上采样回原始大小
        importance_map = F.interpolate(
            block_means.unsqueeze(0).unsqueeze(0), 
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # 归一化到[0,1]
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
        
        # 保存重要性图
        self.importance_maps[camera_id] = importance_map.detach().clone()
        
        # 记录日志
        #logging.info(f"计算完成 camera_id: {camera_id} 的重要性图")
        
        return importance_map



    def update_point_potential_in(self, points_2d, importance_map, camera_id, width, height):
        """
        Enhanced version of update_point_potential that considers surrounding areas when calculating point potentials.
        Uses multiple sampling points around each point to create a more comprehensive potential value.
        
        Args:
            points_2d: 2D coordinates of points
            importance_map: Importance map tensor
            camera_id: Camera identifier
            width: Image width
            height: Image height
        """
        if points_2d.numel() == 0:
            logging.warning(f"camera_id: {camera_id} 的 points_2d 为空")
            return
        
        # Normalize coordinates
        points_2d_normalized = torch.zeros_like(points_2d, device=points_2d.device)
        points_2d_normalized[:, 0] = 2.0 * points_2d[:, 0] / (width - 1) - 1.0
        points_2d_normalized[:, 1] = 2.0 * points_2d[:, 1] / (height - 1) - 1.0
        points_2d_normalized = torch.clamp(points_2d_normalized, -1.0, 1.0)
        
        # Define sampling offsets for surrounding areas (3x3 grid)
        offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],  [0, 0],  [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ], device=points_2d.device) * 0.02  # 0.02 is the sampling radius, can be adjusted
        
        # Create sampling points for each point (including surrounding areas)
        num_points = points_2d_normalized.shape[0]
        num_samples = offsets.shape[0]
        
        # Expand points and offsets for broadcasting
        expanded_points = points_2d_normalized.unsqueeze(1).expand(-1, num_samples, -1)
        expanded_offsets = offsets.unsqueeze(0).expand(num_points, -1, -1)
        
        # Calculate all sampling positions
        sampling_positions = expanded_points + expanded_offsets
        sampling_positions = torch.clamp(sampling_positions, -1.0, 1.0)
        
        # Reshape for grid_sample
        grid = sampling_positions.view(1, -1, 1, 2)
        
        # Sample from importance map for all positions
        sampled_importance = F.grid_sample(
            importance_map.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        # Reshape samples back to (num_points, num_samples)
        sampled_importance = sampled_importance.view(num_points, num_samples)
        
        # Define weights for different positions (center point has higher weight)
        weights = torch.tensor([
            0.5, 1.0, 0.5,
            1.0, 2.0, 1.0,
            0.5, 1.0, 0.5
        ], device=points_2d.device) / 8.0  # Normalize weights
        
        # Calculate weighted average importance for each point
        point_importance = (sampled_importance * weights).sum(dim=1)
        
        # Apply the transformation to boost medium values
        transformed_importance = transform_potential(point_importance)
        
        if torch.isnan(transformed_importance).any() or torch.isinf(transformed_importance).any():
            logging.error(f"camera_id: {camera_id} 的 transformed_importance 包含无效值")
            return
        
        # Update potentials using transformed values
        if camera_id in self.point_potentials:
            old_potential = self.point_potentials[camera_id]
            old_potential_tensor = old_potential['potential']
            
            if old_potential_tensor.shape != transformed_importance.shape:
                old_num = old_potential_tensor.shape[0]
                new_num = transformed_importance.shape[0]
                
                if new_num > old_num:
                    scale_factor = new_num / old_num
                    resized_potential = F.interpolate(
                        old_potential_tensor.unsqueeze(0).unsqueeze(0),
                        scale_factor=scale_factor,
                        mode='nearest'
                    ).squeeze()
                    resized_potential = resized_potential[:new_num]
                else:
                    indices = torch.linspace(0, old_num-1, new_num, dtype=torch.long, device=old_potential_tensor.device)
                    resized_potential = old_potential_tensor[indices]
                
                updated_potential = 0.7 * resized_potential + 0.3 * transformed_importance
                self.point_potentials[camera_id] = {
                    'potential': updated_potential,
                    'count': old_potential['count'] + 1,
                    'timestamp': time.time()
                }
            else:
                updated_potential = self.momentum * old_potential_tensor + (1 - self.momentum) * transformed_importance
                self.point_potentials[camera_id] = {
                    'potential': updated_potential,
                    'count': old_potential['count'] + 1,
                    'timestamp': time.time()
                }
        else:
            self.point_potentials[camera_id] = {
                'potential': transformed_importance,
                'count': 1,
                'timestamp': time.time()
            }
        
        # Cache cleanup based on the oldest timestamp
        if len(self.point_potentials) > self.cache_size:
            oldest_cam = min(self.point_potentials.keys(),
                            key=lambda k: self.point_potentials[k]['timestamp'])
            del self.point_potentials[oldest_cam]
            # logging.info(f"camera_id: {oldest_cam} 的势能缓存已被删除 (最旧)")
    def update_point_potential(self, points_2d, importance_map, camera_id, width, height):
        """
        Modified version of update_point_potential that uses the transformed importance values.
        The core logic remains the same, but applies the transformation before updating potentials.
        """
        if points_2d.numel() == 0:
            logging.warning(f"camera_id: {camera_id} 的 points_2d 为空")
            return
        
        # Normalize coordinates as before
        points_2d_normalized = torch.zeros_like(points_2d, device=points_2d.device)
        points_2d_normalized[:, 0] = 2.0 * points_2d[:, 0] / (width - 1) - 1.0
        points_2d_normalized[:, 1] = 2.0 * points_2d[:, 1] / (height - 1) - 1.0
        points_2d_normalized = torch.clamp(points_2d_normalized, -1.0, 1.0)
        
        grid = points_2d_normalized.view(1, -1, 1, 2)
        
        # Sample from importance map
        point_importance = F.grid_sample(
            importance_map.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        # Apply the transformation to boost medium values
        transformed_importance = transform_potential(point_importance)
        
        if torch.isnan(transformed_importance).any() or torch.isinf(transformed_importance).any():
            logging.error(f"camera_id: {camera_id} 的 transformed_importance 包含无效值")
            return
        
        # Update potentials using transformed values
        if camera_id in self.point_potentials:
            old_potential = self.point_potentials[camera_id]
            old_potential_tensor = old_potential['potential']
            
            if old_potential_tensor.shape != transformed_importance.shape:
                old_num = old_potential_tensor.shape[0]
                new_num = transformed_importance.shape[0]
                
                if new_num > old_num:
                    scale_factor = new_num / old_num
                    resized_potential = F.interpolate(
                        old_potential_tensor.unsqueeze(0).unsqueeze(0),
                        scale_factor=scale_factor,
                        mode='nearest'
                    ).squeeze()
                    resized_potential = resized_potential[:new_num]
                else:
                    indices = torch.linspace(0, old_num-1, new_num, dtype=torch.long, device=old_potential_tensor.device)
                    resized_potential = old_potential_tensor[indices]
                
                updated_potential = 0.7 * resized_potential + 0.3 * transformed_importance
                self.point_potentials[camera_id] = {
                    'potential': updated_potential,
                    'count': old_potential['count'] + 1,
                    'timestamp': time.time()  # Store the current time as the timestamp
                }
            else:
                updated_potential = self.momentum * old_potential_tensor + (1 - self.momentum) * transformed_importance
                self.point_potentials[camera_id] = {
                    'potential': updated_potential,
                    'count': old_potential['count'] + 1,
                    'timestamp': time.time()  # Update the timestamp with the current time
                }
        else:
            self.point_potentials[camera_id] = {
                'potential': transformed_importance,
                'count': 1,
                'timestamp': time.time()  # Set timestamp when first adding the potential
            }

        # Print a message to indicate the cache has been updated
        # logging.info(f"camera_id: {camera_id} 的势能缓存已成功更新")

        # Cache cleanup based on the oldest timestamp
        if len(self.point_potentials) > self.cache_size:
            # Find the camera with the oldest timestamp
            oldest_cam = min(self.point_potentials.keys(),
                            key=lambda k: self.point_potentials[k]['timestamp'])
            del self.point_potentials[oldest_cam]

    def get_potential(self, camera_id):
        """获取指定相机视角的势能"""
        if camera_id in self.point_potentials:
            return self.point_potentials[camera_id]['potential']
        logging.warning(f"camera_id: {camera_id} 的势能不存在")
        return None

    def update_after_prune(self, prune_mask):
        """更新被删除点后的势能信息
        
        Args:
            prune_mask: 布尔张量, True表示要删除的点
        """
        for cam_id in self.point_potentials:
            potential_info = self.point_potentials[cam_id]
            if potential_info['potential'].shape[0] == prune_mask.shape[0]:
                # 保留未被删除的点的势能
                potential_info['potential'] = potential_info['potential'][~prune_mask]
                logging.info(f"已更新 camera_id: {cam_id} 的势能，删除被剪枝的点")

    def clear_cache(self):
        """清理缓存"""
        self.point_potentials.clear()
        self.importance_maps.clear()
        logging.info("已清理 VisualPotential 的所有缓存")

    def world_to_screen(self, points, full_proj_transform, width, height):
        """
        将世界坐标转换为屏幕坐标
        
        Args:
            points: Tensor of shape (N, 3) in world coordinates.
            full_proj_transform: 4x4 projection matrix (camera_to_clip space).
            width: Camera image width.
            height: Camera image height.
        Returns:
            screen_coords: Tensor of shape (N, 2) in screen coordinates (pixel units).
        """
        # 转换为齐次坐标 (N, 4)
        homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        
        # 应用投影变换 (clip space)
        clip_coords = (homogeneous @ full_proj_transform).squeeze()
        
        # 透视除法 (NDC space [-1, 1]^3)
        ndc_coords = clip_coords[:, :3] / clip_coords[:, 3:4]
        
        # 转换为屏幕坐标 (pixel units)
        screen_x = (ndc_coords[:, 0] + 1) * 0.5 * width
        screen_y = (1 - ndc_coords[:, 1]) * 0.5 * height  # 图像坐标系通常Y轴向下
        
        screen_coords = torch.stack([screen_x, screen_y], dim=-1)
        
        # 记录日志
        logging.info("已将世界坐标转换为屏幕坐标")
        
        return screen_coords

    def save_visual_potential_maps(self, output_dir):
        """
        保存所有缓存的重要性图到指定目录
        
        Args:
            output_dir (str): 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for cam_id, imp_map in self.importance_maps.items():
            # 转换为numpy数组并归一化到[0,1]
            imp_np = imp_map.cpu().numpy()
            imp_np = (imp_np - imp_np.min()) / (imp_np.max() - imp_np.min() + 1e-8)
            
            # 绘制图像
            plt.figure(figsize=(12, 8))
            plt.imshow(imp_np, cmap='viridis', interpolation='bilinear')
            plt.colorbar(label='Potential Intensity', shrink=0.8)
            plt.title(f"Visual Potential Map - Camera {cam_id}", fontsize=14)
            plt.axis('off')
            
            # 保存为PNG
            filename = os.path.join(output_dir, f"camera_{cam_id}_potential.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150, pad_inches=0.1)
            plt.close()
            
            logging.info(f"已保存 camera_id: {cam_id} 的重要性图到 {filename}")


def process_importance_map_horizontal_projection(importance_map, full_proj_transform):
    # 1. 计算逆矩阵（保持不变）
    full_proj_transform_inv = torch.linalg.inv(full_proj_transform)

    # 2. 生成三维点云（保持不变）
    h, w = importance_map.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=importance_map.device, dtype=torch.float32),
        torch.arange(w, device=importance_map.device, dtype=torch.float32),
        indexing="ij"
    )
    x_ndc = 2.0 * x_coords / (w - 1) - 1.0
    y_ndc = 2.0 * y_coords / (h - 1) - 1.0
    z_ndc = torch.zeros_like(x_ndc)
    points_ndc = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1).reshape(-1, 3)

    # 3. 应用逆变换（保持不变）
    homogeneous_points = torch.cat([points_ndc, torch.ones_like(points_ndc[:, :1])], dim=-1)
    world_coords = (homogeneous_points @ full_proj_transform_inv.T)[:, :3]

    # 4. 投影到水平面并映射到图像网格（优化坐标范围）
    x_world, y_world = world_coords[:, 0], world_coords[:, 1]
    
    # 计算归一化参数（避免重复计算）
    x_min, x_max = x_world.min(), x_world.max()
    y_min, y_max = y_world.min(), y_world.max()
    
    # 映射到图像网格并限制边界
    x_img = ((x_world - x_min) / (x_max - x_min) * (w - 1)).round().long().clamp(0, w-1)
    y_img = ((y_world - y_min) / (y_max - y_min) * (h - 1)).round().long().clamp(0, h-1)

    # 5. 使用向量化操作替代循环（核心优化）
    flat_indices = y_img * w + x_img
    valid_mask = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    flat_indices = flat_indices[valid_mask]
    importance_values = importance_map.flatten()[valid_mask]

    # 使用bincount累加重复索引的值
    processed_importance_map = torch.bincount(
        flat_indices, 
        weights=importance_values, 
        minlength=h * w
    ).reshape(h, w).to(importance_map.device)

    # 6. 归一化（保持不变）
    processed_importance_map = (processed_importance_map - processed_importance_map.min()) / (
        processed_importance_map.max() - processed_importance_map.min() + 1e-8
    )
    
    return processed_importance_map

def transform_potential(importance_map):
    """
    Transform the importance map to boost medium values and reduce high values.
    
    Args:
        importance_map: Tensor with values in range [0, 1]
    Returns:
        Transformed importance map with values in range [0, 1]
    """
    # First normalize to ensure input is in [0, 1]
    normalized = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    
    # Define the center point for maximum boost (around 0.5)
    center = 0.6
    
    # Create a bell curve centered at 0.5 that boosts medium values
    # and reduces high values using a modified Gaussian function
    sigma = 0.25  # Controls the width of the bell curve
    boost = torch.exp(-((normalized - center) ** 2) / (2 * sigma ** 2))
    
    # Combine original and boosted values with weighting
    # This preserves some of the original signal while boosting medium values
    alpha = 0.8  # Weight for the boost component
    transformed = (1 - alpha) * normalized + alpha * boost
    
    # Normalize again to ensure output is in [0, 1]
    transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-8)
    
    return transformed