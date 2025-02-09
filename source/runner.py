from torch import nn
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


import torch
from torch.cuda.amp import autocast, GradScaler


def train_epoch(
        model=None,
        optimizer=None,
        criterion=None,
        metric=None,
        dataloader=None,
        device="cpu",
        loss_meter=None,
        score_meter=None,
        dice_loss=None,
        lovasz_loss=None,
        focal_loss=None,
        args=None
):
    loss_meter = AverageMeter()
    loss_meter_ce = AverageMeter()
    loss_meter_lovasz = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    # Initialize AMP scaler
    scaler = GradScaler()

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast():  # Automatic mixed precision context
                outputs = model(x)
                loss_ce = criterion(outputs, y)
                loss_focal = focal_loss(outputs, y)
                loss_lovasz = lovasz_loss(outputs.contiguous(), y)
                w_ce, w_focal, w_lovasz = args.weight_ce_focal_lovasz[0], args.weight_ce_focal_lovasz[1], \
                    args.weight_ce_focal_lovasz[2]
                loss = w_ce * loss_ce + w_lovasz * loss_lovasz + w_focal * loss_focal

            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update the scaler for the next iteration

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            loss_meter_ce.update((loss_ce + loss_focal).cpu().detach().numpy(), n=n)
            loss_meter_lovasz.update(loss_focal.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({metric.name: score_meter.avg})
            logs.update({'ce': loss_meter_ce.avg})
            logs.update({'lovasz': loss_meter_lovasz.avg})

            iterator.set_postfix_str(format_logs(logs))

    return logs


def valid_epoch(
        model=None,
        criterion=None,
        metric=None,
        dataloader=None,
        device="cpu",
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                outputs = model.forward(x)
                loss = criterion(outputs, y)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def disable_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # 将BatchNorm替换为Identity层
            module.training = False  # 禁用训练模式
            module.track_running_stats = False  # 禁用统计更新
            module.running_mean = module.running_mean  # 保持当前的均值
            module.running_var = module.running_var  # 保持当前的方差


def train_epoch_ensemble(
        model=None,
        optimizer=None,
        criterion=None,
        metric=None,
        dataloader=None,
        device="cpu",
        loss_meter=None,
        score_meter=None,
        dice_loss=None,
        lovasz_loss=None,
        focal_loss=None,
        args=None
):
    loss_meter = AverageMeter()
    loss_meter_ce = AverageMeter()
    loss_meter_lovasz = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    # Initialize AMP scaler
    scaler = GradScaler()

    model.to(device).train()
    disable_batchnorm(model.encoder_s2)
    disable_batchnorm(model.encoder_sar)
    with tqdm(dataloader, desc="Train") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            esb = sample["img_esb"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(enabled=bool(args.use_amp)):  # Automatic mixed precision context
                outputs = model(esb, x)
                loss_ce = criterion(outputs, y)
                loss_focal = focal_loss(outputs, y)
                loss_lovasz = lovasz_loss(outputs.contiguous(), y)
                w_ce, w_focal, w_lovasz = args.weight_ce_focal_lovasz[0], args.weight_ce_focal_lovasz[1], \
                    args.weight_ce_focal_lovasz[2]
                loss = w_ce * loss_ce + w_lovasz * loss_lovasz + w_focal * loss_focal

            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update the scaler for the next iteration

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            loss_meter_ce.update((loss_ce + loss_focal).cpu().detach().numpy(), n=n)
            loss_meter_lovasz.update(loss_focal.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({metric.name: score_meter.avg})
            logs.update({'ce': loss_meter_ce.avg})
            logs.update({'lovasz': loss_meter_lovasz.avg})

            iterator.set_postfix_str(format_logs(logs))

    return logs


def valid_epoch_ensemble(
        model=None,
        criterion=None,
        metric=None,
        dataloader=None,
        device="cpu",
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            esb = sample["img_esb"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                outputs = model(esb, x)
                loss = criterion(outputs, y)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs
