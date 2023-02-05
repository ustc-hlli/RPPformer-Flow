# RPPformer-Flow
[ACM MM 2022] Code for "RPPformer-Flow: Relative Position Guided Point Transformer for Scene Flow Estmation"

# Preparation
This project is implemented on Python 3.7.10 and Pytorch 1.9.0.

To run our code, first, install the PointNet2 cpp lib as follows.

```
cd pointnet2
python setup.py install
cd ..
```

Then, we go on to prepare the data. Our model is trained on FlyingThings3D and evaluated on FlyingThings3D and KITTI.
We adopt the same preprocessing method as that used in PointPWC-Net and HPLFlowNet. Our instruction is based on the repos.

For FlyingThings3D, please visit the official website, download "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" of DispNet/FlowNet2.0 dataset subsets and unzip them into the same directory, `RAW_DATA_PATH`. After this, run the following script to generate point cloud data.

```
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

For KITTI, please download KITTI Scene Flow Evaluation 2015 and unzip it into the dictionary `RAW_DATA_PATH`. Then 
