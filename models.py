import time
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointnet2 import pointnet2_utils

from model_utils import index_points_group, index_points_gather
from Transformers import DownSampleAtten, UpSampleAtten, Decoder, MLP

class TransformerSceneFlow(nn.Module):
    def __init__(self):
        super(TransformerSceneFlow, self).__init__()

        flow_nei = 32
        feat_nei = 16

        # encoder downsample and decoder
        self.level0 = MLP(3, [32, 32, 64])
        self.flow0 = Decoder(32, 64, 32, feat_nei, flow_nei)

        self.level1 = DownSampleAtten(2048, feat_nei, 64, 128)
        self.flow1 = Decoder(64, 128, 64, feat_nei, flow_nei)
        
        self.level2 = DownSampleAtten(512, feat_nei, 128, 256)
        self.flow2 = Decoder(128, 256, 128, feat_nei, flow_nei)
        
        self.level3 = DownSampleAtten(128, feat_nei, 256, 256)
        self.flow3 = Decoder(256, 0, 256, feat_nei, flow_nei)
        
        self.level4 = DownSampleAtten(64, feat_nei, 256, 256)
        
        #encoder upsample
        self.upsample4 = UpSampleAtten(feat_nei, 256, 256, 256)
        self.upsample3 = UpSampleAtten(feat_nei, 256, 256, 128)
        self.upsample2 = UpSampleAtten(feat_nei, 128, 128, 64)
        self.upsample1 = UpSampleAtten(feat_nei, 64, 64, 32)
  

    def forward(self, xyz1, xyz2, color1, color2):
        '''
        
        Parameters
        ----------
        xyz1 : [B, N1, 3]
        xyz2 : [B, N2, 3]
        color1 : [B, N1, 3]
        color2 :[B, N2, 3]
        
        Returns
        -------
        flows :[B, 3, N1_l]
        fps_pc1_idxs : [B, N1_l]
        fps_pc2_idxs : [B, N2_l]
        pc1 : [B, 3, N1_l]
        pc2 : [B, 3, N2_l]
            
        '''

        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)
        color2 = color2.permute(0, 2, 1)
        
        #encoder downsample
        feat1_l0_d = self.level0(color1)
        feat2_l0_d = self.level0(color2)

        pc1_l1, feat1_l1_d, fps_idx_pc1_l1 = self.level1(pc1_l0, feat1_l0_d)
        pc2_l1, feat2_l1_d, fps_idx_pc2_l1 = self.level1(pc2_l0, feat2_l0_d)

        pc1_l2, feat1_l2_d, fps_idx_pc1_l2 = self.level2(pc1_l1, feat1_l1_d)
        pc2_l2, feat2_l2_d, fps_idx_pc2_l2 = self.level2(pc2_l1, feat2_l1_d)

        pc1_l3, feat1_l3_d, fps_idx_pc1_l3 = self.level3(pc1_l2, feat1_l2_d)
        pc2_l3, feat2_l3_d, fps_idx_pc2_l3 = self.level3(pc2_l2, feat2_l2_d)

        #deepest global level
        pc1_l4, feat1_l4_d, _ = self.level4(pc1_l3, feat1_l3_d)
        _, feat1_l3_u = self.upsample4(pc1_l3, pc1_l4, feat1_l3_d, feat1_l4_d)

        pc2_l4, feat2_l4_d, _ = self.level4(pc2_l3, feat2_l3_d)
        _, feat2_l3_u = self.upsample4(pc2_l3, pc2_l4, feat2_l3_d, feat2_l4_d)
        

        #encoder upsample        
        _, feat1_l2_u = self.upsample3(pc1_l2, pc1_l3, feat1_l2_d, feat1_l3_u)
        _, feat2_l2_u = self.upsample3(pc2_l2, pc2_l3, feat2_l2_d, feat2_l3_u)

        _, feat1_l1_u = self.upsample2(pc1_l1, pc1_l2, feat1_l1_d, feat1_l2_u)
        _, feat2_l1_u = self.upsample2(pc2_l1, pc2_l2, feat2_l1_d, feat2_l2_u)

        _, feat1_l0_u = self.upsample1(pc1_l0, pc1_l1, feat1_l0_d, feat1_l1_u)
        _, feat2_l0_u = self.upsample1(pc2_l0, pc2_l1, feat2_l0_d, feat2_l1_u)


        #decoder
        _, cost3, flow3 = self.flow3(pc1_l3, pc2_l3, pc1_l4, feat1_l3_u, feat2_l3_u, None, None)

        _, cost2, flow2 = self.flow2(pc1_l2, pc2_l2, pc1_l3, feat1_l2_u, feat2_l2_u, cost3, flow3)

        _, cost1, flow1 = self.flow1(pc1_l1, pc2_l1, pc1_l2, feat1_l1_u, feat2_l1_u, cost2, flow2)

        _, _, flow0 = self.flow0(pc1_l0, pc2_l0, pc1_l1, feat1_l0_u, feat2_l0_u, cost1, flow1)
        
        #output
        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_idx_pc1_l1, fps_idx_pc1_l2, fps_idx_pc1_l3]
        fps_pc2_idxs = [fps_idx_pc2_l1, fps_idx_pc2_l2, fps_idx_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2


def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16], scale = 1.0):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points_gather(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss

if __name__ == "__main__":

    model = TransformerSceneFlow()
    pc1 = torch.rand(1, 8192, 3)
    pc2 = torch.rand(1, 8192, 3)
    color1 = torch.rand(1, 8192, 3)
    color2 = torch.rand(1, 8192, 3)
    
    pred_flows, fps_pc1_idxs, _, _, _ = model(pc1, pc2, color1, color2)

