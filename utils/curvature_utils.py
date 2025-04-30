import torch
import curvature_knn._C as _C
from simple_knn._C import distCUDA2

class CurvatureAnalyzer:
    def __init__(self, k_neighbors=16, curvature_threshold=0.5, batch_size=1000, potential_threshold=0.3):
        self.k_neighbors = k_neighbors
        self.curvature_threshold = curvature_threshold
        self.batch_size = batch_size
        self.potential_threshold = potential_threshold
        self.curvature_cache = {}
        
    @torch.no_grad()
    def compute_knn_batch(self, points, batch_indices):
        """使用simple-knn计算KNN"""
        try:
            # 确保输入是连续的CUDA tensor
            points = points.contiguous()
            batch_indices = batch_indices.contiguous()
            
            # 获取查询点
            query_points = points[batch_indices]
            
            # 使用simple-knn计算平均距离
            mean_dists = distCUDA2(points)
            if mean_dists.isnan().any():
                mean_dists = torch.ones_like(mean_dists)
            radius = mean_dists.mean() * 2.0
            
            # 分批计算距离以节省内存
            sub_batch_size = min(1000, len(batch_indices))
            all_knn_indices = []
            
            for i in range(0, len(batch_indices), sub_batch_size):
                sub_query = query_points[i:i + sub_batch_size]
                
                # 计算距离
                dist = torch.cdist(sub_query, points)
                mask = dist < radius
                dist = torch.where(mask, dist, torch.ones_like(dist) * 1e9)
                
                # 找到最近邻
                k = min(self.k_neighbors + 1, points.shape[0])
                _, sub_knn = torch.topk(dist, k, dim=1, largest=False)
                all_knn_indices.append(sub_knn[:, 1:])  # 移除自身
                
                # 清理临时变量
                del dist, mask
                torch.cuda.empty_cache()
            
            # 合并结果
            knn_indices = torch.cat(all_knn_indices, dim=0)
            return knn_indices
            
        except Exception as e:
            print(f"Error in compute_knn_batch: {str(e)}")
            # 返回一个有效的默认值
            return torch.arange(self.k_neighbors, device=points.device).expand(
                len(batch_indices), self.k_neighbors)
        
    @torch.no_grad()
    def compute_selective_curvature(self, points, potential):
        """只对高势能区域的点计算曲率"""
        try:
            # 使用缓存避免重复计算
            cache_key = points.data_ptr()
            if cache_key in self.curvature_cache:
                return self.curvature_cache[cache_key]
                
            N = points.shape[0]
            all_curvatures = torch.zeros(N, device=points.device)
            
            # 只选择高势能的点
            high_potential_mask = potential > self.potential_threshold
            high_potential_indices = torch.nonzero(high_potential_mask).squeeze()
            
            # 处理空张量的情况
            if high_potential_indices.dim() == 0:
                high_potential_indices = high_potential_indices.unsqueeze(0)
            if len(high_potential_indices) == 0:
                return all_curvatures
                
            # 分批处理高势能点
            for start_idx in range(0, len(high_potential_indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(high_potential_indices))
                batch_indices = high_potential_indices[start_idx:end_idx]
                
                try:
                    # 获取当前批次的KNN索引
                    knn_indices = self.compute_knn_batch(points, batch_indices)
                    if knn_indices is None:
                        continue
                        
                    knn_indices = knn_indices.contiguous().int()
                    
                    # 计算当前批次的曲率
                    batch_points = points[batch_indices]
                    batch_curvatures = _C.computeCurvatureCUDA(
                        batch_points, 
                        knn_indices, 
                        min(self.k_neighbors, knn_indices.shape[1])
                    )
                    
                    # 保存结果
                    all_curvatures[batch_indices] = batch_curvatures
                    
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    continue
                
                # 清理临时变量
                torch.cuda.empty_cache()
                
            # 缓存结果
            self.curvature_cache[cache_key] = all_curvatures
            return all_curvatures
            
        except Exception as e:
            print(f"Error in compute_selective_curvature: {str(e)}")
            return torch.zeros_like(potential)
        
    def get_detail_weights(self, curvatures):
        """基于曲率计算细节权重"""
        try:
            weights = torch.sigmoid((curvatures - self.curvature_threshold) * 10)
            return weights.clamp(0.0, 1.0)
        except Exception as e:
            print(f"Error in get_detail_weights: {str(e)}")
            return torch.ones_like(curvatures) * 0.5
        
    def clear_cache(self):
        """清理缓存"""
        self.curvature_cache.clear()