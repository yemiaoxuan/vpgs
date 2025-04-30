import torch
import torch.nn.functional as F

class DensityDistribution:
    def __init__(self, block_size=(64,32), min_size=16, max_size=64, momentum=0.9, batch_size=1024):
        self.block_size = block_size
        self.min_size = min_size
        self.max_size = max_size
        self.momentum = momentum
        self.batch_size = batch_size
        
        # 使用GPU缓存存储密度图和权重
        self.historical_densities = {}
        self.view_graph = {}
        self.importance_weights = {}
        self.density_cache = {}
        
        # 预计算并缓存Sobel算子到GPU
        self.register_sobel_kernels()
    
    def register_sobel_kernels(self):
        """预计算Sobel算子并存储在GPU上"""
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   device="cuda").float().unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   device="cuda").float().unsqueeze(0).unsqueeze(0)

    @torch.cuda.amp.autocast()
    def compute_density_map(self, points_2d, image_size):
        """使用批处理和缓存优化的密度图计算"""
        H = image_size[0] // self.block_size[0]
        W = image_size[1] // self.block_size[1]
        
        # 使用批处理处理大量点
        density_map = torch.zeros((H, W), device="cuda", dtype=torch.float32)
        
        for i in range(0, points_2d.shape[0], self.batch_size):
            batch_points = points_2d[i:i + self.batch_size]
            
            # 计算每个点所属的block索引
            block_x = (batch_points[:, 0] * W).long().clamp(0, W-1)
            block_y = (batch_points[:, 1] * H).long().clamp(0, H-1)
            block_indices = block_y * W + block_x
            
            # 使用bincount进行并行计数
            batch_counts = torch.bincount(block_indices, minlength=H*W)
            density_map += batch_counts.reshape(H, W)
        
        # 归一化密度图
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        return density_map

    @torch.cuda.amp.autocast()
    def compute_temporal_loss(self, cam_id, current_density):
        """优化的时序一致性loss计算"""
        if cam_id not in self.historical_densities:
            return torch.tensor(0.0, device=current_density.device)
            
        return F.mse_loss(
            current_density,
            self.historical_densities[cam_id].detach()
        )

    @torch.cuda.amp.autocast()
    def compute_consistency_loss(self, current_cam, current_density, all_densities):
        """使用批处理优化的一致性loss计算"""
        if current_cam.uid not in self.view_graph:
            return torch.tensor(0.0, device=current_density.device)
            
        # 批量收集相邻视图的密度图
        neighbor_data = [(all_densities[n_id], weight) 
                        for n_id, weight in self.view_graph[current_cam.uid]
                        if n_id in all_densities]
        
        if not neighbor_data:
            return torch.tensor(0.0, device=current_density.device)
            
        # 批量处理所有neighbor
        neighbor_densities = torch.stack([d[0] for d in neighbor_data])
        weights = torch.tensor([d[1] for d in neighbor_data], device=current_density.device)
        
        # 批量计算loss
        losses = F.mse_loss(
            current_density.expand_as(neighbor_densities),
            neighbor_densities,
            reduction='none'
        ).mean(dim=(1,2))
        
        weighted_loss = (losses * weights).sum() / (weights.sum() + 1e-6)
        
        return weighted_loss

    def compute_density_change_rate(self, cam_id, current_density):
        """计算密度图的变化率"""
        if cam_id not in self.historical_densities:
            return torch.zeros_like(current_density, device=current_density.device)
        
        historical_density = self.historical_densities[cam_id].detach()
        density_change_rate = torch.abs((current_density - historical_density) / (historical_density + 1e-6))
        return density_change_rate

    def compute_density_change_loss(self, cam_id, current_density):
        """计算密度变化率的损失"""
        density_change_rate = self.compute_density_change_rate(cam_id, current_density)
        
        # 可以选择不同的损失形式，例如L1损失
        density_change_loss = torch.mean(torch.abs(density_change_rate))
        
        return density_change_loss

    def update_historical_density(self, cam_id, current_density):
        """优化的历史密度更新"""
        if cam_id not in self.historical_densities:
            self.historical_densities[cam_id] = current_density
        else:
            self.historical_densities[cam_id].mul_(self.momentum).add_(
                current_density * (1 - self.momentum)
            )

    @torch.cuda.amp.autocast()
    def build_view_graph(self, cameras, threshold=0.7):
        """使用批处理优化的视图图构建"""
        batch_size = min(len(cameras), self.batch_size)
        
        for i in range(0, len(cameras), batch_size):
            batch_cameras = cameras[i:i + batch_size]
            
            # 批量计算相机位置
            positions = torch.stack([cam.camera_center for cam in batch_cameras])
            
            # 计算当前批次与所有相机的距离
            all_positions = torch.stack([cam.camera_center for cam in cameras])
            dist_matrix = torch.cdist(positions, all_positions)
            similarities = torch.exp(-dist_matrix / 10.0)
            
            # 使用mask进行并行处理
            valid_pairs = similarities > threshold
            
            for j, cam in enumerate(batch_cameras):
                valid_neighbors = valid_pairs[j].nonzero().squeeze(1)
                if len(valid_neighbors) > 0:
                    self.view_graph[cam.uid] = [
                        (cameras[k].uid, similarities[j, k].item())
                        for k in valid_neighbors if i+j != k
                    ]

    def clear_cache(self):
        """清理缓存"""
        self.importance_weights.clear()
        torch.cuda.empty_cache()
