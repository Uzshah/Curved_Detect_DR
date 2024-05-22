from __future__ import absolute_import, division, print_function
import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json


from pathlib import Path


from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy
from timm.utils import ModelEma

from trainer import Trainer
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='START of CHECKPOINT',
                        help='Checkpoint and resume ')

    # Model parameters
    parser.add_argument('--model_name', default='ReXNetV1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default= 512,
                        type=int, help='images input size')
    parser.add_argument('--data_path', default='/home/ushah/Dataset_DR/org_rgb/Apots/Processed_train', type=str, metavar='DATASET PATH',
                        help='Path to dataset')
    parser.add_argument('--train_csv', default='/home/ushah/Dataset_DR/new_train.csv', type=str, metavar='CSV PATH',
                        help='Path to dataset')
    parser.add_argument('--test_csv', default='/home/ushah/Dataset_DR/new_test.csv', type=str, metavar='CSV PATH',
                        help='Path to dataset')
    parser.add_argument('--val_csv', default='/home/ushah/Dataset_DR/new_val.csv', type=str, metavar='CSV PATH',
                        help='Path to dataset')

    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # Optimizer parameters
    parser.add_argument('--opt', default='Adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='CosineAnnealingWarmRestarts', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    
    parser.add_argument("--load_weights_dir", default=None, type=str, help="folder of model to load")

    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
    parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
    parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser
    
def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
