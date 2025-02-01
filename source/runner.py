import torch
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

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()
            outputs = model.forward(x)
            loss_ce = criterion(outputs, y)
            loss_focal = focal_loss(outputs, y)
            loss_lovasz = lovasz_loss(outputs.contiguous(), y)
            w_ce, w_focal, w_lovasz = args.weight_ce_focal_lovasz[0], args.weight_ce_focal_lovasz[1], args.weight_ce_focal_lovasz[2]
            loss = w_ce * loss_ce + w_lovasz * loss_lovasz + w_focal * loss_focal
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            loss_meter_ce.update((loss_ce + loss_focal).cpu().detach().numpy(), n=n)
            loss_meter_lovasz.update(loss_focal.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)
            # print()
            # print('focal loss:', loss_focal.cpu().detach().numpy(), 'lovasz loss:', loss_lovasz.cpu().detach().numpy())
            # logs.update({'loss': loss_meter.avg})
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


def train_epoch2(
        model=None,
        optimizer=None,
        criterion=None,
        metric=None,
        dataloader=None,
        device="cpu",
        loss_meter=None,
        score_meter=None,
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["z"].to(device)
            n = x.shape[0]

            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({"MSE": loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch2(
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
            y = sample["z"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                outputs = model.forward(x)
                loss = criterion(outputs, y)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)
            logs.update({"MSE": loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs
