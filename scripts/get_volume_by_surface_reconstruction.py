import cv2
import numpy as np
import open3d as o3d
import glob

def calculate_box_dimensions(point_cloud):
    # 计算点云的边长
    min_coord = np.min(point_cloud, axis=0)
    max_coord = np.max(point_cloud, axis=0)
    dimensions = max_coord - min_coord
    box_length = np.max(dimensions)

    # 计算点云的体积
    volume = len(point_cloud) * (box_length / len(point_cloud))**3

    return box_length, volume

# 从RGB图像和深度图像中获取点云数据
def generate_point_cloud(rgb_image, depth_image, intrinsics, depth_scale):
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    depth_image = depth_image.astype(np.float32) * depth_scale
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image),
                                                                    o3d.geometry.Image(depth_image),
                                                                    depth_scale=depth_scale,
                                                                    convert_rgb_to_intensity=False)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    point_cloud.remove_non_finite_points(True)

    return np.asarray(point_cloud.points)

# 根据多帧图像估计相机参数
def estimate_camera_parameters(images):
    pattern_size = (8, 6)  # 棋盘格内角点数目
    square_size = 0.0254  # 棋盘格方格的实际尺寸（以米为单位）

    object_points = []
    image_points = []
    h, w = 0, 0

    # 获取棋盘格的内角点坐标
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            h, w = gray.shape[:2]
            object_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
            object_points[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            image_points.append(corners)

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (w, h), None, None)

    return camera_matrix


if __name__ == '__main__':
    # 求位姿：
    image_folder = '/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/box'
    rgb_image_paths = glob.glob(image_folder + '/*.jpg') 
    rgb_image_paths = ['/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/box/video_clips1.jpg','/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/box/video_clips2.jpg','/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/box/video_clips3.jpg']
    # 从图像序列中读取RGB图像和深度图像
    rgb_images = [cv2.imread(path) for path in rgb_image_paths]

    # 估计相机参数
    camera_matrix = estimate_camera_parameters(rgb_images)

    # # 设置深度图像的缩放因子
    # depth_scale = 1000.0

    # # 获取箱子的RGB图像和深度图像
    # box_rgb_image = cv2.imread('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1.jpg')
    # box_depth_image = cv2.imread('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1_depth.jpg', cv2.IMREAD_UNCHANGED)

    # # 生成点云数据
    # point_cloud = generate_point_cloud(box_rgb_image, box_depth_image, camera_matrix, depth_scale)

    # # 计算箱子的边长和体积
    # box_length, volume = calculate_box_dimensions(point_cloud)
    # print('Box Length:', box_length)
    # print('Box Volume:', volume)
