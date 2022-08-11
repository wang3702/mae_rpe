
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
#print("starting rank:", int(os.environ.get("RANK")))
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')



    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
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
    parser.add_argument('--rank', default=-1, type=int,help="specify the rank of script")

    parser.add_argument("--knn_freq", type=int, default=10, help="report current accuracy under specific iterations")
    parser.add_argument("--knn_batch_size", type=int, default=64, help="default batch size for knn eval")
    parser.add_argument("--knn_neighbor", type=int, default=20, help="nearest neighbor used to decide the labels")

    parser.add_argument("--type",type=int,default=0,help="change different adv mode 0: N adversarial mask token \n"
                                                         "1: only one ad mask token")
    parser.add_argument("--torch",type=int,default=1,help="use torch distributed version or not, default:0")
    parser.add_argument("--adv_lr",type=float, default=1.5e-3, help="adversarial learning rate for training")
    parser.add_argument("--adv_wd",type=float, default=0, help="adversarial learning rate for training")

    parser.add_argument("--mlp_dim",type=int,default=1024,help="mlp dimension between encoder and decoder")
    parser.add_argument("--mlp_layer",type=int,default=1,help="number of mlp layers for projection")

    parser.add_argument("--num_crop",type=int,default=4,help="number of crops for training")
    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')
    parser.add_argument("--scale_min",type=float,default=0.08,help="resized crop scale min")
    parser.add_argument("--scale_max",type=float,default=1.0,help="resized crop scale max")
    parser.add_argument("--decoder_depth",type=int,default=8,help="decoder depth")
    parser.add_argument("--adv_type",type=int,default=0,help="adv type applying to multi crop mae model")
    return parser

def add_weight_decay(model,lr,adv_lr,weight_decay=1e-5,adv_wd=1e-5, skip_list=()):
    decay = []
    no_decay = []
    no_decay2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if "adv" in name:
            no_decay2.append(param)
            print("add track of adv param:",name)
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.,"lr":lr},
        {'params': decay, 'weight_decay': weight_decay,"lr":lr},
        {'params': no_decay2, 'weight_decay': -adv_wd,"lr":-adv_lr,'fix_lr':0},
        ]
def main(gpu, ngpus_per_node,args):
    if args.torch==1:
        misc.init_distributed_mode2(gpu,ngpus_per_node,args)
    elif args.torch==2:
        misc.init_distributed_mode_ddp(gpu,ngpus_per_node,args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    print("check rank: {}".format(misc.get_rank()))
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(args.scale_min,args.scale_max), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    from util.MultiTransform import MultiTransform
    transform_train = MultiTransform(transform_train,args.num_crop)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)
    testdir = os.path.join(args.data_path, 'val')
    traindir = os.path.join(args.data_path, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
    # val_dataset = datasets.ImageFolder(traindir,transform_test)
    val_dataset = datasets.ImageFolder(traindir, transform_test)
    test_dataset = datasets.ImageFolder(testdir, transform_test)

    if  args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_val = %s" % str(sampler_val))
        sampler_test = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_test = %s" % str(sampler_test))

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(val_dataset)
        sampler_test = torch.utils.data.RandomSampler(test_dataset)

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
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
            val_dataset, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=args.knn_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    import models_mae
    model = models_mae.__dict__[args.model](args=args,norm_pix_loss=args.norm_pix_loss,img_size=args.input_size)
    import models_vit_rpe
    select_model_name = str(args.model).replace("mae_","")
    model_test = models_vit_rpe.__dict__[select_model_name](
        num_classes=1000,
        global_pool=False,
        )


    model.to(device)
    model_test.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    args.adv_lr = args.adv_lr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("adversarial lr:%.2e"%args.adv_lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    param_groups = add_weight_decay(model,args.lr,args.adv_lr,args.weight_decay,args.adv_wd)
    print(param_groups)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model_test = torch.nn.parallel.DistributedDataParallel(model_test, device_ids=[args.gpu], find_unused_parameters=True)
        model_test_without_ddp = model_test.module
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    from engine_pretrain import  train_multicrop_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_multicrop_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if epoch%args.knn_freq==0 or epoch + 1 == args.epochs:
            from util.knn_monitor import knn_monitor
            checkpoint_now = model_without_ddp.state_dict()
            from util.pos_embed import interpolate_pos_embed
            new_checkpoint={}
            for key in checkpoint_now:
                if 'decoder' in key:
                    #del checkpoint_now[key]
                    continue
                new_checkpoint[key]=checkpoint_now[key]
            checkpoint_now = new_checkpoint
            msg = model_test_without_ddp.load_state_dict(checkpoint_now, strict=False)
            print("loading to 224 model:",msg)
            knn_test_acc = knn_monitor(model_test, data_loader_val, data_loader_test,
                       global_k=args.knn_neighbor, pool_ops=False,
                       vit_backbone=2)
            print("*%d epoch knn acc %f"%(epoch,knn_test_acc))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    print("initial rank:", int(os.environ.get("RANK")))
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.torch==1 or args.torch==2:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print("local ip: ",local_ip)
       # if args.torch==2:
       #     print("init rank:", int(os.environ.get("RANK")))
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.world_size*ngpus_per_node
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node,  args))
    else:
        main(None,None,args)

