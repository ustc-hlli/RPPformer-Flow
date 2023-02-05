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

For FlyingThings3D, please visit the official website, download "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" of DispNet/FlowNet2.0 dataset subsets and unzip them into the same directory, `RAW_DATA_PATH`. After this, run the following command to generate point cloud data.

```
python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

For KITTI, please download KITTI Scene Flow Evaluation 2015 and unzip it into the dictionary `RAW_DATA_PATH`. Then run the following command to generate point cloud data.

```
python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```

# Training
We use the Flyingthings3D training set to train our model. To begin training, first set `exp_name` and `data_root` in the config file `config_train.yaml`.
If the FlyingThings3D data is saved in 'FT3D_PATH/FlyingThings3D_subset_processed_35m/',  `data_root` should be set as 'FT3D_PATH'.
Then simply run the following command.

```
python my_train.py
```

The logs are saved in 'exp_name/logs/' and the checkoints are saved in 'exp_name/checkpoints/'.

# Evaluation
Similar to the training steps, first set the config file `config_test.yaml` before starting evaluation. The required terms are `pretrain`, `dataset`, and `data_root`. Please set `pretrain` as the path to the pretrained model. We also provide our pretrained model that achieves the performance in our paper in `/pretrian`. For `dataset`, you can choose 'FlyingThings3DSubset' or 'KITTI', and `data_root` is the path to the prepared data.
Then simply run the following command.

```
python my_test.py
```

The quantitative results will be displayed when the evaluation process is finished.

# Acknowledgement
Our code is based on PointPWC-Net and HPLFlowNet.
The PointNet2 cpp lib is from pointnet2 and the KNN implementation is from flownet3d.
Sincere thanks for their excellent works! 
