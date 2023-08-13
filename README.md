# Nerf_3d_reconstruction
基于无人机航拍数据的三维场景重建

部分三维重建的结果如results所示

## 数据集建设
#### 1.需要获得地面的多角度图片
#### 2.然后利用colmap估计位姿（作为模型训练中的一个基准）
#### 3.通过Behindthesences算法来获得航拍图像的深度图
#### 无人机数据集（包含深度图）上传于

## 训练
python train.py configs/Tanks/wurenji.yaml



### 我们的代码借鉴了Nope-nerf,Nerfstudio,Ha-nerf, 感谢开源工作者的分享
