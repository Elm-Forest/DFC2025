import argparse
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import ttach as tta
from PIL import Image
from datasets import tqdm
from torch.utils.data import DataLoader

import source
from source.runner import format_logs

warnings.filterwarnings("ignore")


def data_loader(args):
    # get all image paths with ".tif" and "/labels/" in the path
    img_pths = [f for f in Path(args.data_root).rglob("*.tif") if "sar_images" in str(f)]
    # convert paths to strings (if needed)
    img_pths = [str(f) for f in img_pths]

    print("Total samples      :", len(img_pths))

    test_set = source.dataset.Dataset(img_pths, classes=args.classes, train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader


def valid_epoch(
        model=None,
        dataloader=None,
        device="cpu",
):
    model.to(device).eval()
    logs = {}
    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)

            with torch.no_grad():
                outputs = model.forward(x)
                msk = torch.softmax(outputs[:, :, ...], dim=1)
                msk = msk.cpu().numpy()

            pred = msk.argmax(axis=1).astype("uint8")
            for i, item in enumerate(sample["fn"]):
                filename = os.path.splitext(os.path.basename(item))[0]
                y_pr = cv2.resize(pred[i], (1024, 1024), interpolation=cv2.INTER_NEAREST)
                Image.fromarray(y_pr).save(os.path.join(args.save_results, filename + '.png'))
                # print('Processed file:', filename + '.png')

            iterator.set_postfix_str(format_logs(logs))
    return logs


def test_model(args, model, device):
    # get dataset loaders
    test_loader = data_loader(args)

    valid_hist = []
    logs_valid = valid_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    valid_hist.append(logs_valid)


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # using UNet with EfficientNet-B4 backbone
    model = smp.Unet(
        classes=len(args.classes) + 1,
        in_channels=1,
        activation=None,
        encoder_weights="imagenet",
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )

    # count parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)

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

    print("Number of classes  :", len(args.classes) + 1)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale(scales=[1, 2]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    model.to(device).eval()

    # training model
    test_model(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pretrained', type=str, default='model/SAR_Pesudo_model_s0_CELoss.pth')
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', default="K:/dataset/dfc25/val")
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
