import argparse
import sys 
import os 
import glob 
import time
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm 

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from models import TransformerSceneFlow, multiScaleLoss

import transforms
import datasets
import cmd_args 

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = (item.cuda(non_blocking=True) for item in self.next_data)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
    
def training_process(args, logger, chk_dir):
    model = TransformerSceneFlow()

    if(args.multi_gpu):
        model.cuda()
        model = torch.nn.DataParallel(model) 
    else:
        model.cuda()
        
    if(args.pretrain is None):
        print('Training from scratch')
        logger.info('Training from scratch')
        init_epoch = 0
    else:
        model.module.load_stat_dict(torch.load(args.pretrain))
        print('load checkpoint: %s' % args.pretrain)
        logger.info('Load checkpoint: %s' % args.pretrain)
        init_epoch = args.init_epoch
        
    if(args.optimizer == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    else:
        logger.info('Not implemented optimizer')
        raise ValueError('Not implemented optimizer')
        
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch = init_epoch - 1)
    
    if(args.dataset == 'FlyingThings3DSubset'):
        train_dataset = datasets.__dict__[args.dataset](
            train=True,
            transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points),
            num_points=args.num_points,
            data_root=args.data_root,
            full=args.full
        )
        logger.info('train_dataset: ' + str(train_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )

        val_dataset = datasets.__dict__[args.dataset](
            train=False,
            transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
            num_points=args.num_points,
            data_root=args.data_root
        )
        logger.info('val_dataset: ' + str(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
    else:
        logger.info('Not implemented dataset')
        raise ValueError('Not implemented dataset')
        
    best_epe = np.inf
    
    tik = torch.cuda.Event(enable_timing=True)
    tok = torch.cuda.Event(enable_timing=True)
    
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], args.min_learning_rate)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
                  
        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        
        prefetcher = data_prefetcher(train_loader)
        i = 0
        data = prefetcher.next()
        
        while(data is not None):
            tik.record()
            print('it=', i, '/', len(train_loader))
            
            model = model.train()
            pos1, pos2, norm1, norm2, flow = data

            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
            loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            total_loss += loss.detach().cpu().data * flow.size()[0]
            total_seen += flow.size()[0]
            
            i += 1
            data = prefetcher.next()
            
            tok.record()
            torch.cuda.synchronize()

            print('time=', tik.elapsed_time(tok)/1000.0, 's/it')
            
        scheduler.step()
        
        print('total sample: %d' % total_seen)
        train_loss = total_loss / total_seen
        print('EPOCH %d mean training loss: %f' % (epoch, train_loss))
        logger.info('EPOCH %d mean training loss: %f' % (epoch, train_loss))
        
        eval_epe3d, eval_loss = eval_sceneflow(args, model.eval(), val_loader)
        print('EPOCH %d mean epe3d: %f  mean eval loss: %f' % (epoch, eval_epe3d, eval_loss))
        logger.info('EPOCH %d mean epe3d: %f  mean eval loss: %f' % (epoch, eval_epe3d, eval_loss))
        
        if(eval_epe3d < best_epe):
            best_epe = eval_epe3d
            if(args.multi_gpu):
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, epoch, best_epe))
            print('Save model...')
            logger.info('Save model...')
        print('Best epe: %.5f' % (best_epe))
        logger.info('Best epe: %.5f' % (best_epe))
    return best_epe
    
    
def eval_sceneflow(args, model, loader):
    total_seen = 0
    total_loss = 0
    total_epe = 0

    tik = torch.cuda.Event(enable_timing=True)
    tok = torch.cuda.Event(enable_timing=True)

    prefetcher = data_prefetcher(loader)
    i = 0
    data = prefetcher.next()

    while(data is not None):
        tik.record()
        print('it=', i, '/', len(loader))

        pos1, pos2, norm1, norm2, flow = data

        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
            eval_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

            error = pred_flows[0].permute(0, 2, 1) - flow
            epe3d = torch.norm(error, dim=2).mean()
            
            total_loss += eval_loss.cpu().data * flow.size()[0]
            total_epe += epe3d.cpu().data * flow.size()[0]
            total_seen += flow.size()[0]

        i += 1
        data = prefetcher.next()
        
        tok.record()
        torch.cuda.synchronize()
        print('time=', tik.elapsed_time(tok)/1000.0, 's/it')

    print('total sample: %d' % total_seen)
    mean_eval = total_loss / total_seen
    mean_epe3d = total_epe / total_seen

    return mean_epe3d, mean_eval
       
   
    
if __name__ == '__main__':
    root = os.path.dirname(__file__)
    args = cmd_args.parse_args_from_yaml(os.path.join(root, 'config_train.yaml'))
    
    exp_name = args.exp_name
    exper_dir = Path(os.path.join(root, exp_name))

    chk_dir = exper_dir.joinpath('checkpoints')
    log_dir = exper_dir.joinpath('logs')
    
    exper_dir.mkdir(exist_ok= True)
    chk_dir.mkdir(exist_ok= True)
    log_dir.mkdir(exist_ok= True) 
    
    files_to_save = ['config_train.yaml', 'my_train.py', 'models.py', 'Transformers.py', 'model_utils.py']
    for fname in files_to_save:
        file_dir = os.path.join(root, fname)
        os.system('cp %s %s' % (file_dir, log_dir))
    
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print('BEGIN TRAINING...')
    logger.info('BEGIN TRAINING...')
    logger.info(args)
    
    best_epe = training_process(args, logger, chk_dir)

    print('FINISH TRAINING...')
    logger.info('FINISH TRAINING...')