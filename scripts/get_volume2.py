from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import os.path as osp
import sys

import json
import yaml
import numpy as np
import numpy.linalg as npl
import open3d as o3d
import cv2
# 求深度
def get_depth(image_path):

    image = Image.open(image_path)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1_depth.jpg')
    return depth


def load_params(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    

# 读入深度图和彩色图然后进行操作
def imgdepCb(config_path,image_path):
    config = load_params(config_path)
    K = config["camera_K"]
    sensor_height = float(config["sensor_height"])
    base_height = float(config["base_height"])
    box_min_height = float(config["box_min_height"])
    max_depth = config["max_depth"] * 1000
    verbose = config["verbose"]
    image = cv2.imread(image_path)
    # depth = get_depth(image_path)
    depth= cv2.imread('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1_depth.jpg')
    depth[:, 0:700] = 0
    depth[depth > max_depth] = 0
    
    rois_p = []

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if depth[j][i].all() != 0:
                rois_p.append(0.001 * depth[j][i]*np.array([i, j, 1]))
    print(rois_p[100:])
    rois_p = np.vstack(rois_p)

    rois_p3 = (npl.inv(K) @ rois_p.T).T
    
    rois_p3 = rois_p3[np.where(rois_p3[:, 0] < sensor_height - base_height)]
    rois_p3 = rois_p3[np.where(rois_p3[:, 0] > box_min_height)]

    sorted_rois_p3 = find_depth(rois_p3, 2)

    side = sorted_rois_p3[:10000]
    
    plane_mean_y = np.mean(side, axis=0)[1]

    plane0 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] > plane_mean_y)]
    plane1 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] <= plane_mean_y)]
    
    means_0 = []
    means_1 = []
    for a, k in enumerate(range(int(box_min_height*10), int(10*(sensor_height-base_height)))):
        i = k/10
        plane0_ = plane0[np.where((plane0[:, 0] > i) & (plane0[:, 0] < i+0.1))]
        mix0 = find_depth(plane0_, 1, descending=1)[:10]
        means_0.append(np.mean(mix0, axis=0))

        plane1_ = plane1[np.where((plane1[:, 0] > i) & (plane1[:, 0] < i+0.1))]
        mix1 = find_depth(plane1_, 1)[:10]
        means_1.append(np.mean(mix1, axis=0))
    
    means_0_yz = np.mean(np.vstack(means_0), axis=0)
    means_1_yz = np.mean(np.vstack(means_1), axis=0)

    means_side = np.mean(side, axis=0)
    re = np.vstack([means_1_yz, means_side, means_0_yz])
    re[:, 0] = 1
    write_to_json(re.tolist())
    
def find_depth(ps, axis, descending=0):
    if descending==0:
        return ps[ps[:, axis].argsort()]
    else:
        return ps[ps[:, axis].argsort()][::-1]

def write_to_json(ar):
    dic = {"plane_points": ar}
    with open(osp.join("/cloud/private/huh/scripts/biyelunwen/scale_estimation/scripts/configs/points.json"), 'w') as f:
        json.dump(dic, f, indent=2)





if __name__ == '__main__':
    config_path = '/cloud/private/huh/scripts/biyelunwen/scale_estimation/scripts/configs/plane.yaml'
    image_path = '/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1.jpg'
    # get_depth(image_path)
    imgdepCb(config_path,image_path)