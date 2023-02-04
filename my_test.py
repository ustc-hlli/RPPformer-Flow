import argparse
import sys 
import os 
import time
from tqdm import tqdm 
from pathlib import Path
from collections import defaultdict
import glob 
import logging

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from models import TransformerSceneFlow, multiScaleLoss

import transforms
import datasets
import cmd_args 

def flow_metrics(pred_flow, gt_flow):
    """
    Compute EPE3D, Acc3DS, Acc3DR and Outliers3D.

    Parameters
    ---------- 
    pred_flow : [B, N, 3]
        Estimated flow.
    gt_flow : [B, N, 3]
        Contains ground truth flow.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """
    B = pred_flow.size()[0]
    assert(B == 1)

    # Flow
    gt_flow = gt_flow.cpu().numpy()
    pred_flow = pred_flow.cpu().numpy()
    
    #
    l2_norm = np.linalg.norm(gt_flow - pred_flow, axis= 2) #[B, N]
    EPE3D = l2_norm.mean()

    #
    gt_norm = np.linalg.norm(gt_flow, axis= 2) #[B, N]
    relative_err = l2_norm / (gt_norm + 1e-6) #[B, N]

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(float).mean()       
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def test_procedure(args):
    model = TransformerSceneFlow()

    if args.multi_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model)
        model.module.load_state_dict(torch.load(args.pretrain))
        
    else:
        model.cuda()
        model.load_state_dict(torch.load(args.pretrain))
        
    print('load model %s' % args.pretrain)

    
    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root=args.data_root
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    
    metrics = defaultdict(lambda:list())

    sample_num = 0
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.8):
        pos1, pos2, norm1, norm2, flow = data
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
    
        model = model.eval()
        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            EPE3D, acc3d_strict, acc3d_relax, outlier = flow_metrics(pred_flows[0].permute(0, 2, 1), flow)
            metrics['eval_loss'].append(loss.cpu().data.numpy())
            metrics['epe3d'].append(EPE3D)
            metrics['accs'].append(acc3d_strict)
            metrics['accr'].append(acc3d_relax)
            metrics['outlier'].append(outlier)
            sample_num += flow.size()[0]
        
            
    mean_eval = np.sum(metrics['eval_loss'])/sample_num
    mean_epe3d = np.sum(metrics['epe3d'])/sample_num
    mean_accs = np.sum(metrics['accs'])/sample_num
    mean_accr = np.sum(metrics['accr'])/sample_num
    mean_outlier = np.sum(metrics['outlier'])/sample_num
    
    str_out = 'sample_num: %d, mean eval loss: %f, mean epe3d: %f, mean accS: %f, mean accR: %f, mean outlier: %f' % (sample_num, mean_eval, mean_epe3d, mean_accs, mean_accr, mean_outlier)
    print(str_out)

    

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    args = cmd_args.parse_args_from_yaml(os.path.join(root, 'config_test.yaml'))
    test_procedure(args)