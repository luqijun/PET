import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from mmengine.config import Config, DictAction

# 合并配置，以 config 配置为主
def merge_config(config, args):
    for key, value in vars(args).items():
        if key in config:
            continue
        config[key] = value
    return config

def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # config file
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    if args.cfg:
        config =  Config.fromfile(args.cfg)
        if args.cfg_options is not None:
            config.merge_from_dict(args.cfg_options)
        args = merge_config(config, args)

    formatted_params = "Parameters:\n"
    for key, value in args.items():
        formatted_params += "  {}: {}\n".format(key, value)
    print(formatted_params)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # output directory and log 
    if utils.is_main_process:
        output_dir = os.path.join("./outputs", args.dataset_file, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        run_log_name = os.path.join(output_dir, 'run_log.txt')
        with open(run_log_name, "a") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(formatted_params))
            log_file.write("parameters: {}".format(n_parameters))

    # resume
    start_epoch = 0
    best_mae, best_epoch = 1e8, 0
    best_mse, best_mse_epoch = 1e8, 0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint.get('best_mae', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            best_mse = checkpoint.get('best_mse', 0.0)
            best_mse_epoch = checkpoint.get('best_mse_epoch', 0)

    # training
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        t1 = time.time()
        train_stats = train_one_epoch(args,
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        if utils.is_main_process:
            with open(run_log_name, "a") as log_file:
                log_file.write('\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_mae': best_mae,
                'best_epoch': best_epoch,
                'best_mse': best_mse,
                'best_mse_epoch': best_mse_epoch,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if epoch >= args.eval_start and epoch % args.eval_freq == 0 and epoch >= 0:
            t1 = time.time()
            test_stats = evaluate(model, data_loader_val, device, epoch, None)
            t2 = time.time()

            # output results
            mae, mse = test_stats['mae'], test_stats['mse']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
            if mse < best_mse:
                best_mse_epoch = epoch
                best_mse = mse
            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse,
                  "\n\nbest mae:", best_mae, "best epoch:", best_epoch, "\tbest mse:", best_mse, "best epoch:", best_mse_epoch)
            print("==========================\n")
            if utils.is_main_process():
                with open(run_log_name, "a") as log_file:
                    log_file.write("\nepoch: {}, mae: {}, mse: {}, time: {}, \n\n"
                                   "best mae: {}, best epoch: {}\tbest mse: {}, best epoch: {}".format(
                                                epoch, mae, mse, t2 - t1, best_mae, best_epoch, best_mse, best_mse_epoch))
                                                
                # save best checkpoint
                src_path = output_dir / 'checkpoint.pth'
                if mae == best_mae:
                    dst_path = output_dir / 'best_mae_checkpoint.pth'
                    shutil.copyfile(src_path, dst_path)
                if mse == best_mse:
                    dst_path = output_dir / 'best_mse_checkpoint.pth'
                    shutil.copyfile(src_path, dst_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
