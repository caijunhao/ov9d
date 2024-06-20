import os
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import SmoothL1Loss, L1Loss
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from tensorboardX import SummaryWriter

from models.model import OV9D
from models.optimizer import build_optimizers
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import glob
import utils.utils as utils


def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)
    optimizer = None

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    utils.init_distributed_mode_torchrun(args)
    print(args)
    device = torch.device(args.gpu)

    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, args.data_name, str(args.batch_size), 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.scale_size), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)

    # Logging
    if args.rank == 0:
        exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), exp_name)
        log_dir = os.path.join(args.log_dir, exp_name)
        logging.check_and_make_dirs(log_dir)
        writer = SummaryWriter(logdir=log_dir)
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.log_args_to_txt(log_txt, args)

        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        log_txt = None
        log_dir = None
        
    model = OV9D(args=args)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset, 
        'data_path': args.data_path, 
        'data_name': args.data_name, 
        'data_type': args.data_train, 
    }
    dataset_kwargs['scale_size'] = args.scale_size

    train_dataset = get_dataset(**dataset_kwargs)
    dataset_kwargs['data_type'] = args.data_val
    dataset_kwargs['num_view'] = 50
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=True, 
    )

    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size,
                                               sampler=sampler_train,
                                               num_workers=args.workers, 
                                               pin_memory=True, 
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=1, 
                                             sampler=sampler_val,
                                             pin_memory=True)
    
    # Training settings
    criterion_o = SmoothL1Loss(beta=0.1)
    # criterion_o = L1Loss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1
    if args.resume_from:
        load_model(args.resume_from, model.module, optimizer)
        print(f'resumed from ckpt {args.resume_from}')
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        idx = [int(ckpt.split('/')[-1].split('_')[-2]) for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    # Perform experiment
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_o, log_txt, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        if args.rank == 0:
            writer.add_scalar('Training loss', loss_train, epoch)
        
        if args.rank == 0:
            if args.save_model:
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(log_dir, 'last.ckpt'))
        
        loss_val = validate(val_loader, model, criterion_o, 
                            device=device, epoch=epoch, args=args)

        if args.rank == 0:
            writer.add_scalar('Validation loss', loss_val, epoch)

        if args.rank == 0:
            torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                    },
                    os.path.join(log_dir, f'epoch_{epoch}_model.ckpt'))
    


def train(train_loader, model, criterion_o, log_txt, optimizer, device, epoch, args):    
    global global_step
    model.train()
    nocs_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step /
                                            iterations/half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step /
                                            iterations/half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale']

        input_RGB = batch['image'].to(device)
        input_MASK = batch['mask'].to(device).to(bool)
        nocs = batch['nocs'].to(device).permute(0, 2, 3, 1)
        dis_sym = batch['dis_sym'].to(device)
        con_sym = batch['con_sym'].to(device)

        preds = model(input_RGB, class_ids=batch['class_id'])
        pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)
        
        pred_nocs_list, gt_nocs_list = [], []
        for b in range(batch['image'].shape[0]):
            curr_pred_nocs = pred_nocs[b]
            curr_gt_nocs = nocs[b]
            curr_mask = input_MASK[b]
            curr_pred_nocs = curr_pred_nocs[curr_mask]
            curr_gt_nocs = curr_gt_nocs[curr_mask]
            curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
            # discrete symmetry
            curr_dis_sym = dis_sym[b]
            dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
            curr_dis_sym = curr_dis_sym[dis_sym_flag]
            aug_pcl_m = torch.stack([curr_pcl_m], dim=0)
            for sym in curr_dis_sym:
                rot, t = sym[0:3, 0:3], sym[0:3, 3]
                rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
                aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0)
            # continuous symmetry
            curr_con_sym = con_sym[b]
            con_sym_flag = torch.sum(torch.abs(curr_con_sym), dim=(-1)) != 0
            curr_con_sym = curr_con_sym[con_sym_flag]
            for sym in curr_con_sym:
                axis = sym[:3].cpu().numpy()
                angles = np.deg2rad(np.arange(5, 180, 5))
                rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
                rots = torch.from_numpy(R.from_rotvec(rotvecs).as_matrix()).to(curr_pcl_m)
                rot_pcl_m_list = []
                for rot in rots:
                    rot_pcl_m = aug_pcl_m @ rot.T
                    rot_pcl_m_list.append(rot_pcl_m)
                aug_pcl_m = torch.cat([aug_pcl_m] + rot_pcl_m_list, dim=0)
            curr_gt_nocs_set = aug_pcl_m + 0.5
            with torch.no_grad():
                curr_gt_nocs_set = torch.unbind(curr_gt_nocs_set, dim=0)
                loss = list(map(lambda gt_nocs: criterion_o(curr_pred_nocs, gt_nocs), curr_gt_nocs_set))
                min_idx = torch.argmin(torch.tensor(loss))
            curr_gt_nocs = curr_gt_nocs_set[min_idx]
            
            pred_nocs_list.append(curr_pred_nocs)
            gt_nocs_list.append(curr_gt_nocs)

        optimizer.zero_grad()
        loss_o = criterion_o(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list))

        nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))
        loss_o.backward()
        
        if args.rank == 0:
            if batch_idx % args.print_freq == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Loss: {loss}, Mov Avg Loss: {ma_loss}, LR: {lr}\n'.format(
                        epoch, batch_idx, iterations,
                        loss=loss_o, 
                        ma_loss=nocs_loss.avg, 
                        lr=current_lr, 
                    )
                result_lines.append(result_line)
                print(result_line)
        optimizer.step()
    
    if args.rank == 0:
        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            for result_line in result_lines:
                txtfile.write(result_line)   

    return nocs_loss.avg


def validate(val_loader, model, criterion_o, device, epoch, args):    
    model.eval()
    nocs_loss = logging.AverageMeter()
    iterations = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):      

        with torch.no_grad():
            input_RGB = batch['image'].to(device)
            input_MASK = batch['mask'].to(device).to(bool)
            nocs = batch['nocs'].to(device).permute(0, 2, 3, 1)
            dis_sym = batch['dis_sym'].to(device)
            con_sym = batch['con_sym'].to(device)

            preds = model(input_RGB, class_ids=batch['class_id'])
            pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)
            
            pred_nocs_list, gt_nocs_list = [], []
            for b in range(batch['image'].shape[0]):
                curr_pred_nocs = pred_nocs[b]
                curr_gt_nocs = nocs[b]
                curr_mask = input_MASK[b]
                curr_pred_nocs = curr_pred_nocs[curr_mask]
                curr_gt_nocs = curr_gt_nocs[curr_mask]
                curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
                # discrete symmetry
                curr_dis_sym = dis_sym[b]
                dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
                curr_dis_sym = curr_dis_sym[dis_sym_flag]
                aug_pcl_m = torch.stack([curr_pcl_m], dim=0)
                for sym in curr_dis_sym:
                    rot, t = sym[0:3, 0:3], sym[0:3, 3]
                    rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
                    aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0)
                # continuous symmetry
                curr_con_sym = con_sym[b]
                con_sym_flag = torch.sum(torch.abs(curr_con_sym), dim=(-1)) != 0
                curr_con_sym = curr_con_sym[con_sym_flag]
                for sym in curr_con_sym:
                    axis = sym[:3].cpu().numpy()
                    angles = np.deg2rad(np.arange(5, 180, 5))
                    rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
                    rots = torch.from_numpy(R.from_rotvec(rotvecs).as_matrix()).to(curr_pcl_m)
                    rot_pcl_m_list = []
                    for rot in rots:
                        rot_pcl_m = aug_pcl_m @ rot.T
                        rot_pcl_m_list.append(rot_pcl_m)
                    aug_pcl_m = torch.cat([aug_pcl_m] + rot_pcl_m_list, dim=0)
                curr_gt_nocs_set = aug_pcl_m + 0.5
                with torch.no_grad():
                    curr_gt_nocs_set = torch.unbind(curr_gt_nocs_set, dim=0)
                    loss = list(map(lambda gt_nocs: criterion_o(curr_pred_nocs, gt_nocs), curr_gt_nocs_set))
                    min_idx = torch.argmin(torch.tensor(loss))
                curr_gt_nocs = curr_gt_nocs_set[min_idx]
                
                pred_nocs_list.append(curr_pred_nocs)
                gt_nocs_list.append(curr_gt_nocs)

            loss_o = criterion_o(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list))

        nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))
        
        if args.rank == 0:
            if batch_idx % (args.print_freq * 10) == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Val Loss: {loss} Mov Avg Val Loss {ma_loss}\n'.format(
                        epoch, batch_idx, iterations,
                        loss=loss_o, 
                        ma_loss=nocs_loss.avg, 
                    )
                print(result_line)
    
    return nocs_loss.avg


if __name__ == '__main__':
    main()
