import os
import os.path as osp
import sys

import json
import yaml
import numpy as np
import numpy.linalg as npl
import open3d as o3d
import math
import time
import toml
import socket
import cv2
from get_depth import get_depth

plane_config_path= '/cloud/private/huh/scripts/biyelunwen/scale_estimation/scripts/configs/plane.yaml'


class FitPlane:
    def __init__(self):
        self.config = self.load_params(plane_config_path)
        self.K = self.config["camera_K"]
        self.sensor_height = float(self.config["sensor_height"])
        self.base_height = float(self.config["base_height"])
        self.box_min_height = float(self.config["box_min_height"])

        self.max_depth = self.config["max_depth"] * 1000
        self.verbose = self.config["verbose"]


    def load_params(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    
    # 读入深度图和彩色图然后进行操作
    def imgdepCb(self):
        image_path = '/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1_depth.jpg'
        image = cv2.imread(image_path)
        #depth = cv2.imread('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1_depth.jpg')
        depth = get_depth(image_path)
        depth[:, 0:700] = 0
        depth[depth > self.max_depth] = 0
        
        rois_p = []
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                print(depth[i][j])
                if depth[j][i] != 0:
                    rois_p.append(0.001 * depth[j][i]*np.array([i, j, 1]))

        rois_p = np.vstack(rois_p)

        rois_p3 = (npl.inv(self.K) @ rois_p.T).T
        
        rois_p3 = rois_p3[np.where(rois_p3[:, 0] < self.sensor_height - self.base_height)]
        rois_p3 = rois_p3[np.where(rois_p3[:, 0] > self.box_min_height)]

        sorted_rois_p3 = self.find_depth(rois_p3, 2)
 
        side = sorted_rois_p3[:10000]
        
        plane_mean_y = np.mean(side, axis=0)[1]

        plane0 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] > plane_mean_y)]
        plane1 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] <= plane_mean_y)]
        
        means_0 = []
        means_1 = []
        for a, k in enumerate(range(int(self.box_min_height*10), int(10*(self.sensor_height-self.base_height)))):
            i = k/10
            plane0_ = plane0[np.where((plane0[:, 0] > i) & (plane0[:, 0] < i+0.1))]
            mix0 = self.find_depth(plane0_, 1, descending=1)[:10]
            means_0.append(np.mean(mix0, axis=0))

            plane1_ = plane1[np.where((plane1[:, 0] > i) & (plane1[:, 0] < i+0.1))]
            mix1 = self.find_depth(plane1_, 1)[:10]
            means_1.append(np.mean(mix1, axis=0))
        
        means_0_yz = np.mean(np.vstack(means_0), axis=0)
        means_1_yz = np.mean(np.vstack(means_1), axis=0)

        means_side = np.mean(side, axis=0)
        re = np.vstack([means_1_yz, means_side, means_0_yz])
        re[:, 0] = 1
        self.write_to_json(re.tolist())
        
    def find_depth(self, ps, axis, descending=0):
        if descending==0:
            return ps[ps[:, axis].argsort()]
        else:
            return ps[ps[:, axis].argsort()][::-1]

    def write_to_json(self, ar):
        dic = {"plane_points": ar}
        with open(osp.join("/cloud/private/huh/scripts/biyelunwen/scale_estimation/scripts/configs/points.json"), 'w') as f:
            json.dump(dic, f, indent=2)






if __name__ == '__main__':

    fit = FitPlane()
    fit.imgdepCb()
    
