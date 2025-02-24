import argparse
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from timm.optim import Adan
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torch.utils.data import DataLoader

import source
from source.dataset import Dataset_Fusion
from source.fusion_model import Efficient_UAF
from source.lovasz_losses import LovaszLoss
from source.polyloss import Poly1FocalLoss

warnings.filterwarnings("ignore")


def data_loader(args):
    # 获取所有标签路径
    img_pths = [f for f in Path(args.data_root).rglob("*.tif") if "labels" in str(f)]
    # 打乱路径顺序
    random.shuffle(img_pths)
    # 按比例划分训练集和验证集
    split_idx = int(args.train_proportion * len(img_pths))
    train_pths = img_pths[:split_idx]
    val_pths = img_pths[split_idx:]
    # 转换为字符串路径
    train_pths = [str(f) for f in train_pths]
    val_pths = [str(f) for f in val_pths]

    print("Total samples      :", len(img_pths))
    print("Training samples   :", len(train_pths))
    print("Validation samples :", len(val_pths))

    # 创建训练和验证数据集
    trainset = Dataset_Fusion(train_pths, ensemble_folders=args.ensemble_folders, classes=args.classes,
                              size=args.crop_size, train=True)
    validset = Dataset_Fusion(val_pths, ensemble_folders=args.ensemble_folders, classes=args.classes,
                              size=args.val_size, train=False)

    # 创建DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size_val, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 如果使用分布式训练
    train_sampler = None
    if torch.cuda.device_count() > 1 and args.use_ddp == 1:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(trainset)
        train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)

    return train_loader, valid_loader, train_sampler


def train_model(args, model, optimizer, criterion, metric, device):
    # get dataset loaders
    train_data_loader, val_data_loader, train_sampler = data_loader(args)

    # initialize learning rate scheduler
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    uint = args.n_epochs // args.lr_cycle
    scheduler = CosineLRScheduler(optimizer=optimizer,
                                  t_initial=uint,
                                  cycle_limit=args.lr_cycle,
                                  cycle_mul=0.95,
                                  cycle_decay=0.8,
                                  lr_min=1e-6,
                                  warmup_t=args.warmup_epochs,
                                  warmup_lr_init=args.warmup_lr)
    # create folder to save model
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_checkpoint, exist_ok=True)
    model_name = f"Binary_{args.save_model}_s{args.seed}"
    # dice_loss = DiceLoss().to(device)
    # focal_loss = FocalLoss(alpha=args.focal_alpha_gamma[0],
    #                        gamma=args.focal_alpha_gamma[1],
    #                        label_smoothing=0.1,
    #                        reduction='mean').to(device)

    focal_loss = Poly1FocalLoss(num_classes=2,
                                label_is_onehot=True,
                                alpha=args.focal_alpha_gamma[0],
                                gamma=args.focal_alpha_gamma[1],
                                reduction='mean').to(device)
    lovasz_loss = LovaszLoss(mode='binary').to(device)
    max_score = 0
    train_hist = []
    valid_hist = []
    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch + 1}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        logs_train = source.runner.train_epoch_ensemble(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_data_loader,
            device=device,
            dice_loss=None,
            lovasz_loss=lovasz_loss,
            focal_loss=focal_loss,
            args=args
        )

        if (epoch + 1) % args.save_checkpoint_ep == 0:
            torch.save(model.state_dict(), os.path.join(args.save_checkpoint, f"checkpoint_ep{epoch}_{model_name}.pth"))
            print("Checkpoint saved in the folder : ", args.save_checkpoint)

        logs_valid = source.runner.valid_epoch_ensemble(
            model=model,
            criterion=criterion,
            metric=metric,
            dataloader=val_data_loader,
            device=device,
            args=args
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)
        score = logs_valid[metric.name]

        # # necessary print
        # print('CE weights:', criterion.weight)

        if max_score < score:
            max_score = score
            torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}.pth"))
            print("Model saved in the folder : ", args.save_model)
            print("Model name is : ", model_name)

        # Step the scheduler at the end of each epoch
        scheduler.step(epoch)


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(os.getenv('LOCAL_RANK', -1))  # 确保转换为整数
    if torch.cuda.device_count() > 1 and args.use_ddp == 1:
        print("Parallel training!")
        # os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # using UNet with EfficientNet-B4 backbone
    # model = smp.Unet(
    #     classes=len(args.classes) + 1,
    #     in_channels=1,
    #     activation=None,
    #     encoder_weights="imagenet",
    #     encoder_name="efficientnet-b4",
    #     decoder_attention_type="scse",
    # )

    model = Efficient_UAF(in_channels_sar=1,
                          in_channels_single=1,
                          ensemble_num=len(args.ensemble_folders),
                          encoder_name=args.encoder_name,
                          classes=len(args.classes) + 1)

    # model = SOLCV7(num_classes=len(args.classes) + 1,
    #                in_ch_sar=1,
    #                in_ch_opt=len(args.ensemble_folders))
    # count parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)

    classes_wt = np.ones([len(args.classes) + 1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt).to(device)
    metric = source.metrics.IoU2()
    model.to(device)
    optimizer = Adan(model.parameters(),
                     lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    # optimizer = torch.optim.NAdam(model.parameters(),
    #                               lr=args.learning_rate,
    #                               weight_decay=args.weight_decay)
    if args.pretrained is not None:
        print("Loading weights...")
        weights = torch.load(args.pretrained, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(torch.load(args.pretrained))
            print('Pretrained Loading success!')
        except:
            new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print('loading success after replace module')
            except Exception as inst:
                print('pass loading weights')
                print(inst)

    if torch.cuda.device_count() > 1 and args.use_ddp == 1:
        print("Using DDP")
        print("Number of GPUs :", torch.cuda.device_count())
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank, find_unused_parameters=True)
        optimizer = Adan(params=model.module.parameters(),
                         lr=args.learning_rate,
                         weight_decay=args.weight_decay)

    elif torch.cuda.device_count() > 1:
        print("Using DP")
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        optimizer = Adan(params=model.module.parameters(),
                         lr=args.learning_rate,
                         weight_decay=args.weight_decay)

    print("Number of epochs    :", args.n_epochs)
    print("Number of classes   :", len(args.classes) + 1)
    print("Batch size          :", args.batch_size)
    print("Device              :", device)
    print("Focal loss          :", args.focal_alpha_gamma)

    # training model
    train_model(args, model, optimizer, criterion, metric, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--model_name', default="deeplab")
    parser.add_argument('--encoder_name', default="efficientnet-b4")
    parser.add_argument('--model_size', default="b4")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batch_size_val', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_proportion', type=float, default=0.9)
    parser.add_argument('--fiter_threshold', type=float, default=0.01)
    parser.add_argument('--class_weights', type=float, nargs='*', default=[1.0, 19.0])
    parser.add_argument('--focal_alpha_gamma', type=float, nargs='*', default=[0.25, 2.0])
    parser.add_argument('--weight_ce_focal_lovasz', type=float, nargs='*', default=[0.35, 0.5, 0.15])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--val_size', type=int, default=512)
    parser.add_argument('--change_size', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_cycle', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--use_ddp', type=int, default=0)
    parser.add_argument('--use_amp', type=int, default=1)
    parser.add_argument('--test_eval', type=int, default=0)
    parser.add_argument('--use_binary_aug', action="store_true")
    parser.add_argument('--classes', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6, 7, 8])
    # parser.add_argument('--ensemble_folders', type=str, nargs='*', default=[
    #     'ensemble_files/uper_mitb5_0305',
    #     'ensemble_files/segf_mitb4_0299',
    #     'ensemble_files/uper_cnext2_0307',
    #     'ensemble_files/uper_mitb5_0302',
    #     'ensemble_files/uper_convnext2L_0315'
    # ])
    parser.add_argument('--ensemble_folders', type=str, nargs='*', default=[
        'ensemble_train',
        # 'K:/dataset/dfc25/train/sar_images',
        # 'K:/dataset/dfc25/train/sar_images',
        # 'K:/dataset/dfc25/train/sar_images',
        # 'K:/dataset/dfc25/train/sar_images'
    ])
    parser.add_argument('--data_root', default="K:/dataset/dfc25/train")
    parser.add_argument('--save_model', default="model_fusion")
    parser.add_argument('--save_checkpoint', default="checkpoints")
    parser.add_argument('--save_checkpoint_ep', type=int, default=5)
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
