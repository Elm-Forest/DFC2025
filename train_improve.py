import argparse
import random
import time
import warnings
import argparse
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

import source

def data_loader(args):
    # get all image paths with ".tif" and "/labels/" in the path
    img_pths = [f for f in Path(args.data_root).rglob("*.tif") if "labels" in str(f)]
    # Shuffle the paths to randomize the selection
    random.shuffle(img_pths)
    # split data: 90% training and 10% validation
    split_idx = int(args.train_proportion * len(img_pths))
    train_pths = img_pths[:split_idx]
    val_pths = img_pths[split_idx:]
    # convert paths to strings (if needed)
    train_pths = [str(f) for f in train_pths]
    val_pths = [str(f) for f in val_pths]

    print("Total samples      :", len(img_pths))
    print("Training samples   :", len(train_pths))
    print("Validation samples :", len(val_pths))

    train_set = source.dataset.Dataset(train_pths, classes=args.classes, size=args.crop_size, train=True)
    valid_set = source.dataset.Dataset(val_pths, classes=args.classes, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers)

    return train_loader, valid_loader


def main(args):
    args


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batch_size_val', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_proportion', type=float, default=0.9)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
    parser.add_argument('--lr_step', type=int, default=2, help='lr decay rate')
    parser.add_argument('--lr_start_epoch_decay', type=int, default=1, help='epoch to start lr decay')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', default="K:/dataset/dfc25/train")
    parser.add_argument('--save_model', default="model")
    parser.add_argument('--save_results', default="results")
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--load_weights', type=bool, default=True)
    parser.add_argument('--weights_path', type=str, default='checkpoint/checkpoint_xiaorong_no2_9.pth')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint")
    parser.add_argument('--frozen', type=bool, default=False)
    parser.add_argument('--input_channels', type=int, default=13)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
