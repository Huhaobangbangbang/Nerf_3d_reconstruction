import subprocess
import numpy as np
import glob
def estimate_camera_matrix(image_paths):
    # 使用OpenMVG进行SFM
    command = ['openMVG_main_SfMInit_ImageListing', '-i', ','.join(image_paths), '-o', 'sfm_output']
    subprocess.call(command)

    # 读取相机参数
    camera_params = {}
    with open('sfm_output/cameras.txt', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            camera_id, params = line.strip().split()
            camera_params[int(camera_id)] = [float(param) for param in params.split(',')]

    # 构建相机矩阵
    camera_matrix = np.zeros((3, 3))
    camera_matrix[0, 0] = camera_params[0][0]
    camera_matrix[1, 1] = camera_params[0][1]
    camera_matrix[0, 2] = camera_params[0][2]
    camera_matrix[1, 2] = camera_params[0][3]
    camera_matrix[2, 2] = 1.0

    return camera_matrix

# 示例使用：




image_folder = '/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/box'
image_paths = glob.glob(image_folder + '/*.jpg') 
camera_matrix = estimate_camera_matrix(image_paths)
camera_matrix = estimate_camera_matrix(image_paths)

print('Camera Matrix:')
print(camera_matrix)
print('Camera Matrix:')
print(camera_matrix)
