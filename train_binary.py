import argparse
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader

import source
from source.focal_loss import FocalLoss
from source.lovasz_losses import LovaszLoss
from source.model import creatModel

warnings.filterwarnings("ignore")


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

    trainset = source.dataset.Dataset_limit(train_pths, classes=args.classes, size=args.crop_size, train=True,
                                            use_binary=True, cls_id=args.classes[0],
                                            fiter_threshold=args.fiter_threshold)
    validset = source.dataset.Dataset_limit(val_pths, classes=args.classes, train=False, use_binary=False,
                                            cls_id=args.classes[0], fiter_threshold=args.fiter_threshold)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(validset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers)

    return train_loader, valid_loader


def train_model(args, model, optimizer, criterion, metric, device):
    # get dataset loaders
    train_data_loader, val_data_loader = data_loader(args)

    # initialize learning rate scheduler
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    scheduler = CosineLRScheduler(optimizer=optimizer,
                                  t_initial=args.n_epochs,
                                  lr_min=5e-6,
                                  warmup_t=args.warmup_epochs,
                                  warmup_lr_init=args.warmup_lr)
    # create folder to save model
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_checkpoint, exist_ok=True)
    model_name = f"Binary_{args.save_model}_s{args.seed}_{criterion.name}"
    # dice_loss = DiceLoss().to(device)
    focal_loss = FocalLoss(alpha=args.focal_alpha_gamma[0], gamma=args.focal_alpha_gamma[1], reduction='mean').to(
        device)
    lovasz_loss = LovaszLoss(mode='binary').to(device)
    max_score = 0
    train_hist = []
    valid_hist = []
    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch + 1}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        logs_train = source.runner.train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_data_loader,
            device=device,
            dice_loss=None,
            lovasz_loss=lovasz_loss,
            focal_loss=focal_loss,
        )

        if (epoch + 1) % args.save_checkpoint_ep == 0:
            torch.save(model.state_dict(), os.path.join(args.save_checkpoint, f"checkpoint_ep{epoch}_{model_name}.pth"))
            print("Checkpoint saved in the folder : ", args.save_checkpoint)

        logs_valid = source.runner.valid_epoch(
            model=model,
            criterion=criterion,
            metric=metric,
            dataloader=val_data_loader,
            device=device,
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)
        score = logs_valid[metric.name]

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

    # using UNet with EfficientNet-B4 backbone
    # model = smp.Unet(
    #     classes=len(args.classes) + 1,
    #     in_channels=1,
    #     activation=None,
    #     encoder_weights="imagenet",
    #     encoder_name="efficientnet-b4",
    #     decoder_attention_type="scse",
    # )

    model = creatModel(args)

    # count parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)

    classes_wt = np.array(args.class_weights)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)
    metric = source.metrics.IoU2()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay,
                                 fused=True)
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

    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            [dict(params=model.module.parameters(), lr=args.learning_rate)]
        )

    print("Number of epochs    :", args.n_epochs)
    print("Number of classes   :", len(args.classes) + 1)
    print("Batch size          :", args.batch_size)
    print("Device              :", device)
    print("Class weights       :", args.class_weights)
    print("Focal loss          :", args.focal_alpha_gamma)
    print("Class id            :", args.classes)

    # training model
    train_model(args, model, optimizer, criterion, metric, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--warmup_lr', type=float, default=5e-5)
    parser.add_argument('--model_name', default="deeplab")
    parser.add_argument('--encoder_name', default="mit_b4")
    parser.add_argument('--model_size', default="b4")
    parser.add_argument('--batch_size', type=int, default=3)
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--classes', type=int, nargs='*', default=[4])
    parser.add_argument('--data_root', default="K:/dataset/dfc25/train")
    parser.add_argument('--save_model', default="Binary_model")
    parser.add_argument('--save_checkpoint', default="checkpoints")
    parser.add_argument('--save_checkpoint_ep', type=int, default=5)
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
