import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import  FSC147
from models import models_mae_cross


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=26, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--resume', default='./output_pre_4_dir/checkpoint-300.pth',
    #                    help='resume from checkpoint')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = FSC147(args.data_path, split = "train")
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    loss_scaler = NativeScaler()

    min_MAE = 99999

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # train one epoch
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        accum_iter = args.accum_iter

        # some parameters in training
        train_mae = 0
        train_rmse = 0
        pred_cnt = 0
        gt_cnt = 0

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))
        
        for data_iter_step, (samples, gt_density, boxes, m_flag) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):

            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

            samples = samples.to(device, non_blocking=True).half()
            gt_density = gt_density.to(device, non_blocking=True).half()
            boxes = boxes.to(device, non_blocking=True).half()

            # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
            flag = 0
            for i in range(m_flag.shape[0]):
                flag += m_flag[i].item()
            if flag == 0:
                shot_num = random.randint(0,3)
            else:
                shot_num = random.randint(1,3)

            with torch.cuda.amp.autocast():
                output = model(samples,boxes,shot_num)

            # Compute loss function
            mask = np.random.binomial(n=1, p=0.8, size=[384,384])
            masks = np.tile(mask,(output.shape[0],1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(device)
            loss = (output - gt_density) ** 2
            loss = (loss * masks / (384*384)).sum() / output.shape[0]

            loss_value = loss.item()

            # Update information of MAE and RMSE
            batch_mae = 0
            batch_rmse = 0
            for i in range(output.shape[0]):
                pred_cnt = torch.sum(output[i]/60).item()
                gt_cnt = torch.sum(gt_density[i]/60).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2

                if i == 0 :
                    print(f'{data_iter_step}/{len(data_loader_train)}: loss: {loss_value},  pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {abs(pred_cnt - gt_cnt)},  AE: {cnt_err},  SE: {cnt_err ** 2}, {shot_num}-shot ')

            train_mae += batch_mae
            train_rmse += batch_rmse
                    
            # Output visualisation information to tensorboard
            if log_writer is not None and data_iter_step == 0:
                fig = output[0].unsqueeze(0).repeat(3,1,1)
                f1 = gt_density[0].unsqueeze(0).repeat(3,1,1)

                log_writer.add_images('bboxes', (boxes[0]), int(epoch),dataformats='NCHW')
                log_writer.add_images('gt_density', (samples[0]/2+f1/10), int(epoch),dataformats='CHW')
                log_writer.add_images('density map', (fig/20), int(epoch),dataformats='CHW')
                log_writer.add_images('density map overlay', (samples[0]/2+fig/10), int(epoch),dataformats='CHW')

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
                        
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            # if log_writer is not None and (data_iter_step + 1) % accum_iter == 3:
            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('MAE', batch_mae/args.batch_size, epoch_1000x)
                log_writer.add_scalar('RMSE', (batch_rmse/args.batch_size)**0.5, epoch_1000x)

        # Only use 1 batches when overfitting
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 

        # save train status and model
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and train_mae/(len(data_loader_train) * args.batch_size) < min_MAE:
            min_MAE = train_mae/(len(data_loader_train) * args.batch_size)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=666)

        
        # Output log status
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'Current MAE': train_mae/(len(data_loader_train) * args.batch_size),
                        'RMSE':  (train_rmse/(len(data_loader_train) * args.batch_size))**0.5,
                        'epoch': epoch,}

        print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format( train_mae/(len(data_loader_train) * args.batch_size), (train_rmse/(len(data_loader_train) * args.batch_size))**0.5))

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)