import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import knn_point, index_points_gather, index_points_group as index_points
from model_utils import UpsampleFlow
from pointnet2 import pointnet2_utils

USE_GN = True


class Conv1d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, use_gn=USE_GN):
        super(Conv1d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.composed_module = nn.Sequential(
                                nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.GroupNorm(out_feat//16, out_feat) if(use_gn) else nn.Identity(),
                                nn.LeakyReLU(0.1, inplace=True)
                                )

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat= [], use_gn= USE_GN):
        super(MLP, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.layers = nn.ModuleList()
        
        last_feat = in_feat
        for feat in out_feat:
            self.layers.append(Conv1d(last_feat, feat))
            last_feat = feat
        
    def forward(self, feat):
        for layer in self.layers:
            feat = layer(feat)
            
        return feat

class RPPAttenLayer(nn.Module):
    def __init__(self, in_feat, out_feat, nsample, skip_connection= True, use_gn= USE_GN):
        super(RPPAttenLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.nsample = nsample
        self.skip_connection = skip_connection
        
        self.pos_encoder=nn.Sequential(
                                    nn.Conv1d(in_channels= 3, out_channels= in_feat, kernel_size= 1),
                                    nn.GroupNorm(in_feat//16, in_feat) if(use_gn) else nn.Identity(),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    )
        self.early_k_encoder=nn.Sequential( 
                                    nn.Conv2d(in_channels= 3, out_channels= in_feat, kernel_size= 1),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        self.early_q_encoder=nn.Sequential( 
                                    nn.Conv2d(in_channels= 3, out_channels= in_feat, kernel_size= 1),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        self.early_v_encoder_1=nn.Sequential(
                                    nn.Conv2d(in_channels= 3, out_channels= in_feat, kernel_size= 1),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        self.early_v_encoder_2=nn.Sequential(
                                    nn.Conv2d(in_channels= 3, out_channels= in_feat, kernel_size= 1),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        self.qk_encoder=nn.Conv2d(in_channels= in_feat, out_channels= (in_feat), kernel_size= 1, bias=False)
        self.v_encoder=nn.Conv2d(in_channels= in_feat, out_channels= (in_feat), kernel_size= 1, bias=False)
        self.feed_forward=nn.Sequential(
                                    nn.Conv1d(in_channels= in_feat, out_channels= out_feat, kernel_size= 1),
                                    nn.GroupNorm(out_feat//16, out_feat) if(use_gn) else nn.Identity(),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        
        
    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        
        Parameters
        ----------
        xyz : [B,3,N]
        feat : [B,C,N]

        Returns
        -------
        xyz1 : [B,3,N1]
        feat_new : [B,C_n,N1]

        """
    
        B, _, N1 = xyz1.size()
        N2 = xyz2.size()[2]
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()
        
        pos_code1 = self.pos_encoder(xyz1) #[B, C, N1]
        feat1 = feat1 + pos_code1 #[B, C, N1]
        pos_code2 = self.pos_encoder(xyz2) 
        feat2 = feat2 + pos_code2
        
        idx = knn_point(self.nsample, xyz2_t, xyz1_t)
        xyz2_sampled = index_points(xyz2_t, idx).permute(0, 3, 1, 2) #[B, 3, N1, S]
        #_, idx = pointnet2_utils.knn(self.nsample, xyz1_t, xyz2_t)
        #xyz2_sampled = pointnet2_utils.grouping_operation(xyz2.contiguous(), idx)
        real_xyz = xyz2_sampled- xyz1.view(B, -1, N1, 1) #[B, 3, N1, S]
                
        early_k_code = self.early_k_encoder(real_xyz) #[B, C, N1, S]
        early_q_code = self.early_q_encoder(real_xyz) 
        early_v_code_1 = self.early_v_encoder_1(real_xyz)
        early_v_code_2 = self.early_v_encoder_2(real_xyz)
        
        feat1_real = feat1.view(B, -1, N1, 1) #[B, C, N1, 1(S)]
        feat2_real = index_points(feat2.permute(0, 2, 1), idx).permute(0, 3, 1, 2) #[B, C, N1, S]
        #feat2_real = pointnet2_utils.grouping_operation(feat2.contiguous(), idx)
          
        q = self.qk_encoder(feat1_real.mul(early_q_code)) #[B, C, N1, S] 
        k = feat2_real.mul(early_k_code) 
        v = self.v_encoder(feat2_real.mul(early_v_code_1)).mul(early_v_code_2) #[B, C, N1, S]


        atten = torch.einsum('bcns, bcns -> bns', q, k) #[B, N1, S]
        atten = torch.softmax(atten/np.sqrt(k.size()[1]), dim=-1)
        atten = atten.view(B, 1, N1, self.nsample).repeat(1, self.in_feat, 1, 1)
        
        feat_new = torch.einsum('bcns, bcns -> bcn', v, atten) #[B, C, N1]
        if(self.skip_connection):
            feat_new = feat_new + feat1
        feat_new = self.feed_forward(feat_new)
        
        return xyz1, feat_new
              
class CostBlock(nn.Module):
    def __init__(self, in_feat, out_feat, nsample, use_gn= USE_GN):
        super(CostBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.nsample = nsample

        self.inter_att = RPPAttenLayer(in_feat=self.in_feat, out_feat=self.in_feat, nsample=self.nsample)
        self.intra_att = RPPAttenLayer(in_feat=self.in_feat, out_feat=self.out_feat, nsample=self.nsample)
        
    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        
        Parameters
        ----------
        xyz : [B,3,N]
        feat : [B,C,N]

        Returns
        -------
        xyz1 : [B,3,N1]
        flow_embedding : [B,C_n,N1]

        """

        _, point_cost = self.inter_att(xyz1, xyz2, feat1, feat2)
        _, patch_cost = self.intra_att(xyz1, xyz1, point_cost, point_cost)

        return xyz1, patch_cost

class DownSampleAtten(nn.Module):
    def __init__(self, npoints, nsample, in_feat, out_feat, use_gn= USE_GN):
        super(DownSampleAtten, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.npoints = npoints
        self.nsample = nsample
        
        self.att = RPPAttenLayer(self.in_feat, self.out_feat, self.nsample)

    def forward(self, xyz, feat):
        """
        
        Parameters
        ----------
        xyz : [B,3,N]
        feat : [B,C,N]

        Returns
        -------
        new_xyz : [B,3,N']
        new_feat : [B,C',N']

        """
        
        B, _, N = xyz.size()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        
        fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoints) #[B, N']
        new_xyz = index_points_gather(xyz_t, fps_idx).permute(0, 2, 1) #[B, 3, N']
        new_feat = index_points_gather(feat.permute(0, 2, 1), fps_idx).permute(0, 2, 1) #[B, C, N']
        #new_xyz =  pointnet2_utils.gather_operation(xyz.contiguous(), fps_idx)        
        #new_feat = pointnet2_utils.gather_operation(feat.contiguous(), fps_idx)

        _, new_feat = self.att(new_xyz, xyz, new_feat, feat)        
        
        return new_xyz, new_feat, fps_idx

class UpSampleAtten(nn.Module):
    def __init__(self, nsample, in_feat_1, in_feat_2, out_feat, use_gn= USE_GN):
        super(UpSampleAtten, self).__init__()
        self.in_feat_1 = in_feat_1
        self.in_feat_2 = in_feat_2
        self.out_feat = out_feat
        self.nsample = nsample
        
        self.pre_forward= nn.Sequential(
                                nn.Conv1d(in_channels= in_feat_2, out_channels= in_feat_1, kernel_size= 1),
                                nn.GroupNorm(in_feat_1//16, in_feat_1) if(use_gn) else nn.Identity(),
                                nn.LeakyReLU(0.1, inplace=True)
                                )
        self.att= RPPAttenLayer(self.in_feat_1, self.out_feat, self.nsample)

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        
        Parameters
        ----------
        xyz : [B,3,N]
        feat : [B,C,N]

        Returns
        -------
        new_xyz : [B,3,N']
        new_feat : [B,C',N']

        """
        
        B, _, N1 = xyz1.size()
       
        feat2 = self.pre_forward(feat2)
        _, upsampled_feat = self.att(xyz1, xyz2, feat1, feat2)        
        
        return xyz1, upsampled_feat

class Decoder(nn.Module):
    def __init__(self, in_feat_point, in_feat_cost, out_feat, nneibor, nsample, use_gn= USE_GN):
        super(Decoder, self).__init__()
        self.in_feat_point = in_feat_point
        self.in_feat_cost = in_feat_cost
        self.out_feat = out_feat
        self.nneibor = nneibor
        self.nsample = nsample
        
        self.upsampler = UpsampleFlow()
        self.att = CostBlock(in_feat=self.in_feat_point, out_feat=self.out_feat, nsample=self.nsample)
        self.estimator = nn.Sequential(nn.Conv1d(in_channels=self.out_feat, out_channels=64, kernel_size=1),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1))
        if(self.in_feat_cost != 0):
            self.upsample_att = UpSampleAtten(self.nneibor, self.out_feat, self.in_feat_cost, self.out_feat)
    
    def forward(self, xyz1, xyz2, xyz1d, feat1, feat2, costd, flowd):
        B, _, N1=xyz1.size()
        
        
        if(flowd is not None):
            upsampled_flow = self.upsampler(xyz1, xyz1d, flowd)
            warped_xyz1 = xyz1 + upsampled_flow
            warped_xyz1d = xyz1d + flowd
        else:
            warped_xyz1 = xyz1
            warped_xyz1d = xyz1d
        
        _, cost= self.att(warped_xyz1, xyz2, feat1, feat2)
        
        if(costd is not None):
            _, cost = self.upsample_att(warped_xyz1, warped_xyz1d, cost, costd)

        new_flow = self.estimator(cost)
        
        if(flowd is not None):
            new_flow = new_flow + upsampled_flow
        
        return xyz1, cost, new_flow