import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

def load_point_cloud(file_path):
    # 加载点云文件
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def detect_density_geometric_keypoints(pcd, radius=0.1, density_threshold=30, curvature_threshold=0.04, nms_radius=0.15):
    """
    基于点云密度和几何特征的关键点检测

    参数:
        pcd: open3d点云对象
        radius: 搜索半径
        density_threshold: 密度阈值，用于筛选高密度区域
        curvature_threshold: 曲率阈值，用于筛选特征明显的点
    """
    # 将点云转换为numpy数组
    points = np.asarray(pcd.points)

    # 1. 计算局部密度
    kdtree = KDTree(points)
    densities = []
    for point in points:
        neighbors = kdtree.query_ball_point(point, radius)
        densities.append(len(neighbors))
    densities = np.array(densities)

    # 2. 计算局部几何特征
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)
    curvatures = []

    for i, point in enumerate(points):
        neighbors = kdtree.query_ball_point(point, radius)
        if len(neighbors) < 3:
            curvatures.append(0)
            continue

        # 计算局部协方差矩阵
        neighbor_points = points[neighbors]
        centered_points = neighbor_points - point
        cov = np.cov(centered_points.T)

        # 计算特征值
        if cov.shape == (3, 3):  # 确保有足够的点计算协方差
            eigenvalues = np.linalg.eigvals(cov)
            eigenvalues = np.sort(np.abs(eigenvalues))
            # 使用最小特征值与总和的比率作为曲率估计
            curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
            curvatures.append(curvature)
        else:
            curvatures.append(0)

    curvatures = np.array(curvatures)

    # 3. 综合考虑密度和几何特征选择关键点
    keypoint_indices = []
    for i in range(len(points)):
        # 同时满足密度和曲率条件的点被选为关键点
        if (densities[i] > density_threshold and
                curvatures[i] > curvature_threshold):
            keypoint_indices.append(i)
    keypoint_indices = []
    is_keypoint = np.zeros(len(points), dtype=bool)

    # 按照特征响应度（密度 * 曲率）排序
    response = densities * curvatures
    sorted_indices = np.argsort(-response)  # 降序排序

    for idx in sorted_indices:
        if response[idx] < density_threshold * curvature_threshold:
            continue

        # 检查邻域内是否已有关键点
        neighbors = kdtree.query_ball_point(points[idx], nms_radius)
        if not any(is_keypoint[neighbors]):
            keypoint_indices.append(idx)
            is_keypoint[idx] = True

    # 创建关键点点云
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(points[keypoint_indices])

    print(f"检测到的关键点数量: {len(keypoint_indices)}")
    return keypoints

def visualize_separated_point_clouds(original_pcd, keypoints):
    # 平移关键点云
    keypoint_cloud = o3d.geometry.PointCloud()
    keypoint_cloud.points = keypoints.points
    keypoint_cloud.paint_uniform_color([1, 0, 0])  # 红色

    # 平移原始点云和关键点云（避免重叠）
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    keypoint_cloud.translate((10, 0, 0))  # 将关键点云沿x轴平移

    # 同时可视化
    o3d.visualization.draw_geometries([original_pcd, keypoint_cloud], window_name="Separated Point Clouds")

# def main():
#     # 1. 加载点云
#     ply_file = "room.ply"  # 替换为您的 PLY 文件路径
#     pcd = load_point_cloud(ply_file)
#     print("点云加载完成，点数:", len(pcd.points))

#     # 2. 检测关键点（可以调整参数）
#     keypoints = detect_density_geometric_keypoints(
#         pcd,
#         radius=0.12, density_threshold=100, curvature_threshold=0.06,nms_radius=0.3
#     )

#     # 3. 可视化结果

#     visualize_separated_point_clouds(pcd, keypoints)

#     # 4. 输出关键点坐标（可选）
#     keypoint_coordinates = np.asarray(keypoints.points)
#     print("\n前5个关键点的坐标:")
#     print(keypoint_coordinates[:5])

# if __name__ == "__main__":
#     main()
