import argparse
import math
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import ttach as tta
from PIL import Image

import source
from source.model import creatModel

warnings.filterwarnings("ignore")

# class palette
class_rgb = {
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

# class labels
class_gray = {
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}


def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]

    return out


def test_model(args, model, device):
    # path to save predictions
    os.makedirs(args.save_results, exist_ok=True)
    # load test data
    test_fns = [f for f in Path(args.data_root).rglob("*.tif") if "sar_images" in str(f)]
    for fn_img in test_fns:
        img = source.dataset.load_grayscale(fn_img)
        h, w = img.shape[:2]
        power = math.ceil(np.log2(h) / np.log2(2))
        shape = (2 ** power, 2 ** power)
        img = cv2.resize(img, shape)

        # test time augmentation
        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, ::-1].copy())
        imgs.append(img[::-1, :].copy())
        imgs.append(img[::-1, ::-1].copy())

        input_ = torch.cat([TF.to_tensor(x).unsqueeze(0) for x in imgs], dim=0).float().to(device)
        pred = []
        with torch.no_grad():
            msk = model(input_)
            msk = torch.softmax(msk[:, :, ...], dim=1)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1]) / 4
        pred = pred.argmax(axis=0).astype("uint8")

        y_pr = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        # save image as png
        filename = os.path.splitext(os.path.basename(fn_img))[0]
        res = y_pr
        # res = label2rgb(y_pr)
        Image.fromarray(res).save(os.path.join(args.save_results, filename + '.png'))
        print('Processed file:', filename + '.png')
    print("Done!")
    print("Total files processed: ", len(test_fns))


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # n_classes = len(args.classes) + 1
    model = creatModel(args)

    if args.pretrained_model is not None:
        print("Loading weights...")
        weights = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(torch.load(args.pretrained_model))
            print('Pretrained Loading success!')
        except:
            new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print('loading success after replace module')
            except Exception as inst:
                print('pass loading weights')
                print(inst)
    transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
            # tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            # tta.Scale(scales=[0.8, 1, 1.2]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    model.to(device).eval()

    # test model
    test_model(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--model_name', default="uper")
    parser.add_argument('--encoder_name', default="mit_b5")
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', default="K:/dataset/dfc25/val")
    parser.add_argument('--pretrained_model', default="model/SAR_Pesudo_uper_mitb5_exp_ep40_337.pth")
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
