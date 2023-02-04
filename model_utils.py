import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2 import pointnet2_utils

KNN_MODE_POINTNET = True

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz, pointnet=KNN_MODE_POINTNET):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, K]
    """
    if(pointnet):
        _, group_idx = pointnet2_utils.knn(nsample, new_xyz.contiguous(), xyz.contiguous())
    else:
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)

    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)

    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

class UpsampleFlow(nn.Module):  
    def forward(self, xyz, sparse_xyz, sparse_flow):
        '''
        3-nn inverse-distance weighted interpolation
        Inputs:
        xyz: coordinates of target points [B, 3, N]
        sparse_xyz: coordinates of source point [B, 3, S]
        
        '''
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) #[B, N, 3]
        sparse_xyz = sparse_xyz.permute(0, 2, 1) #[B, S, 3]
        sparse_flow = sparse_flow.permute(0, 2, 1) #[B, S, 3]
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C) #[B, N, 3(S), 3]
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) #[B, N, 3]
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm  #[B, N, 3]

        grouped_flow = index_points_group(sparse_flow, knn_idx)  #[B, N, 3, C]
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1) #[B, 3, N]

        return dense_flow 