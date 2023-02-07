# RPPformer-Flow
**[ACM MM 2022] Code for "RPPformer-Flow: Relative Position Guided Point Transformer for Scene Flow Estmation"**  
*Hanlin Li∗, Guanting Dong∗, Yueyi Zhang†, Xiaoyan Sun, Zhiwei Xiong*  

![image of RPP Attention Layer](https://github.com/ustc-hlli/RPPformer-Flow/blob/main/images/img.jpg)

## Preparation
This project is implemented with Python 3.7.10 and Pytorch 1.9.0.


To run our code, first, install the PointNet2 cpp lib as follows.

```
cd pointnet2
python setup.py install
cd ..
```


Then, we go on to prepare the data. Our model is trained on FlyingThings3D and evaluated on FlyingThings3D and KITTI.
We adopt the same preprocessing method as that used in [PointPWC-Net](https://github.com/DylanWusee/PointPWC) and [HPLFlowNet](https://github.com/laoreja/HPLFlowNet). Our instruction is based on the repos.

For FlyingThings3D, please visit the official [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), download "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" of DispNet/FlowNet2.0 dataset subsets and unzip them into the same directory, `RAW_DATA_PATH`. After this, run the following command to generate point cloud data.

```
python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

For KITTI, please download [KITTI Scene Flow Evaluation 2015](https://www.cvlibs.net/download.php?file=data_scene_flow.zip) and unzip it into the dictionary `RAW_DATA_PATH`. Then run the following command to generate point cloud data.

```
python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```

## Training
We use the Flyingthings3D training set to train our model. To begin training, first set `exp_name` and `data_root` in the config file `config_train.yaml`.
If the FlyingThings3D data is saved in `FT3D_PATH/FlyingThings3D_subset_processed_35m/`, then `data_root` should be set as 'FT3D_PATH'.
Then simply run the following command.

```
python my_train.py
```

The logs are saved in `exp_name/logs/` and the checkoints are saved in `exp_name/checkpoints/`.

## Evaluation
Similar to the training steps, first set the config file `config_test.yaml` before starting evaluation. The required terms are `pretrain`, `dataset`, and `data_root`. Please set `pretrain` as the path to the pretrained model. We also provide our pretrained model that achieves the performance in our paper in `/pretrian`. For `dataset`, you can choose 'FlyingThings3DSubset' or 'KITTI', and `data_root` is the path to the prepared data.
Then simply run the following command.

```
python my_test.py
```

The quantitative results will be displayed when the evaluation process is finished.

## Citation
If you use this project in your academic work, please cite as:

```
@inproceedings{li2022rppformer,
  title={RPPformer-Flow: Relative Position Guided Point Transformer for Scene Flow Estimation},
  author={Li, Hanlin and Dong, Guanting and Zhang, Yueyi and Sun, Xiaoyan and Xiong, Zhiwei},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4867--4876},
  year={2022}
}
```

## Acknowledgement
Our code is based on [PointPWC-Net](https://github.com/DylanWusee/PointPWC) and [HPLFlowNet](https://github.com/laoreja/HPLFlowNet).
The PointNet2 cpp lib is from the [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch) and the KNN implementation is from the [repo](https://github.com/hyangwinter/flownet3d_pytorch).

Sincere thanks for their excellent work! 
