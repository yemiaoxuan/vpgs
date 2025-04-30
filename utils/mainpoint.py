import open3d as o3d
import numpy as np
from scipy.spatial import KDTree


def load_point_cloud_and_attributes(ply_file, curvature_file, normals_file):
    """
    加载点云文件、曲率文件和法向量文件
    """
    print(f"Loading ply file: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)

    print(f"Loading curvature file: {curvature_file}")
    curvatures = np.loadtxt(curvature_file)  # 加载曲率文件

    print(f"Loading normals file: {normals_file}")
    normals = np.loadtxt(normals_file)  # 加载法向量文件

    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 验证数据维度匹配
    assert len(pcd.points) == len(curvatures), "点云点数与曲率数量不匹配"
    assert len(pcd.points) == len(normals), "点云点数与法向量数量不匹配"

    return pcd, curvatures, normals


def detect_density_geometric_keypoints(pcd, curvatures, radius=0.1, density_threshold=100, curvature_threshold=0.05, nms_radius=0.2, max_keypoints=160):
    points = np.asarray(pcd.points)
    
    # 加载曲率数据
    if isinstance(curvatures, str):
        curvatures = np.loadtxt(curvatures)
    if curvatures.ndim == 2:
        curvatures = np.linalg.norm(curvatures, axis=1)
    assert len(points) == len(curvatures)

    # 1. 使用Open3D的KDTree加速密度计算
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = np.zeros(len(points), dtype=int)
    for i in range(len(points)):
        densities[i] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)[0]

    # 2. 计算响应值并预过滤低响应点
    response = densities * curvatures
    min_response = density_threshold * curvature_threshold
    valid_indices = np.where(response >= min_response)[0]
    sorted_indices = valid_indices[np.argsort(-response[valid_indices])]

    # 3. 基于向量化距离计算的NMS
    current_nms_radius = nms_radius
    keypoint_indices = []

    while True:
        selected_points = []
        candidate_indices = []
        
        for idx in sorted_indices:
            pt = points[idx]
            # 向量化距离检查
            if len(selected_points) > 0:
                pts_array = np.array(selected_points)
                dists_sq = np.sum((pts_array - pt)**2, axis=1)
                if np.any(dists_sq < current_nms_radius**2):
                    continue
            selected_points.append(pt)
            candidate_indices.append(idx)
            if len(candidate_indices) >= max_keypoints:
                break

        # 动态调整半径逻辑
        if len(candidate_indices) <= max_keypoints or current_nms_radius > radius*100:
            keypoint_indices = candidate_indices
            break
            
        current_nms_radius *= 1.5
        print(f"Adjusting nms_radius to {current_nms_radius:.2f}, current keypoints: {len(candidate_indices)}")

    # 创建关键点点云
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(points[keypoint_indices])
    print(f"Detected {len(keypoint_indices)} keypoints")
    return keypoints



def visualize_separated_point_clouds(original_pcd, keypoints):
    """
    分别显示原始点云和关键点
    """
    keypoint_cloud = o3d.geometry.PointCloud()
    keypoint_cloud.points = keypoints.points
    keypoint_cloud.paint_uniform_color([1, 0, 0])  # 红色

    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    keypoint_cloud.translate((0, 50, 0))  # 将关键点云沿x轴平移

    o3d.visualization.draw_geometries([original_pcd, keypoint_cloud], window_name="Separated Point Clouds")


# def main(ply_file, curvature_file, normals_file, radius=0.1, density_threshold=50, curvature_threshold=0.05, nms_radius=0.2):
#     """
#     主函数：加载点云、计算关键点并显示
#     """
#     print("加载点云和属性...")
#     pcd, curvatures, _ = load_point_cloud_and_attributes(ply_file, curvature_file, normals_file)

#     print("检测关键点...")
#     keypoints = detect_density_geometric_keypoints(pcd, curvatures, radius, density_threshold, curvature_threshold, nms_radius)

#     print("分别显示原始点云和关键点...")
#     visualize_separated_point_clouds(pcd, keypoints)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="基于密度和几何特征的点云关键点检测")
#     parser.add_argument("input_ply", type=str, help="输入PLY文件路径")
#     parser.add_argument("curvature_file", type=str, help="输入曲率文件路径")
#     parser.add_argument("normals_file", type=str, help="输入法向量文件路径")
#     parser.add_argument("--radius", type=float, default=0.1, help="局部密度计算搜索半径")
#     parser.add_argument("--density_threshold", type=int, default=100, help="密度阈值")
#     parser.add_argument("--curvature_threshold", type=float, default=0.4, help="曲率阈值")
#     parser.add_argument("--nms_radius", type=float, default=5.0, help="非极大值抑制半径")

#     args = parser.parse_args()

#     main(args.input_ply, args.curvature_file, args.normals_file,
#          args.radius, args.density_threshold, args.curvature_threshold, args.nms_radius)




