import warnings

import numpy as np
import torchvision.transforms.functional as TF

# reference: https://albumentations.ai/

warnings.simplefilter("ignore")


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) for v in self.classes]
        msk = np.stack(msks, axis=-1).astype(np.float32)
        background = 1 - msk.sum(axis=-1, keepdims=True)
        sample["mask"] = TF.to_tensor(np.concatenate((background, msk), axis=-1))

        for key in [k for k in sample.keys() if k != "mask"]:
            sample[key] = TF.to_tensor(sample[key].astype(np.float32) / 255.0)
        return sample


def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def test_augm(sample):
    augms = [A.Flip(p=0.1)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


import albumentations as A


def train_augm(sample, size=512):
    augms = [
        # 旋转、平移、缩放
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5
        ),

        # 随机选择 Resize 或 RandomCrop
        A.OneOf([
            A.Resize(height=size, width=size, p=1.0),
            A.RandomSizedCrop(min_max_height=(400, 700), size=(size, size), p=1.0),
            # A.RandomCrop(height=size, width=size, p=1.0)
        ], p=1.0),

        # D4 dihedral group transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),

        # 下采样
        A.Downscale(scale_range=(0.5, 0.75), p=0.05),

        # 保留变形和噪声相关增强
        A.OneOf(
            [
                # 弹性变换（Elastic Transform）
                A.ElasticTransform(p=1),

                # 光学畸变（Optical Distortion）
                A.OpticalDistortion(p=1),

                # 网格畸变（Grid Distortion）
                A.GridDistortion(p=1),

                # 透视变换（Perspective Transform）
                A.Perspective(p=1),
            ],
            p=0.1,
        ),

        A.OneOf([
            A.CLAHE(p=1),
            A.Sharpen(p=1),
            A.RandomGamma(gamma_limit=(60, 120), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ],
            p=0.3
        ),

        # 噪声增强
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.SaltAndPepper(p=1),
                # Hard
                A.Superpixels(p=1),

                A.GaussianBlur(p=1),
                A.AdvancedBlur(p=1.0),
                A.MotionBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),
    ]

    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm_binary(sample, size=512):
    augms = [
        # # 旋转、平移、缩放
        # A.ShiftScaleRotate(
        #     scale_limit=0, rotate_limit=45, border_mode=0, p=1
        # ),
        A.Resize(height=size, width=size, p=1.0),  # 统一大小，不丢失信息
        # 水平翻转
        A.HorizontalFlip(p=0.5),
        # 垂直翻转
        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.1),
        # 下采样
        A.Downscale(scale_range=(0.5, 0.75), p=0.05),

        # 保留变形和噪声相关增强
        A.OneOf(
            [
                # 弹性变换（Elastic Transform）
                A.ElasticTransform(p=1),

                # 光学畸变（Optical Distortion）
                A.OpticalDistortion(p=1),

                # 网格畸变（Grid Distortion）
                A.GridDistortion(p=1),

                # 透视变换（Perspective Transform）
                # A.Perspective(p=1),  # p=1表示一定会应用透视变换
            ],
            p=0.1,
        ),

        # 噪声增强
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.Sharpen(p=1),
                A.GaussianBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ],
            p=0.2,
        ),
        # A.Normalize(normalization='standard'),
    ]

    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm3(sample, size=512):
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def valid_augm2(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms, additional_targets={'osm': 'image'})(image=sample["image"], mask=sample["mask"],
                                                                 osm=sample["osm"])


def train_augm2(sample, size=512):
    augms = [
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7
        ),
        A.RandomCrop(size, size, p=1.0),
        A.Flip(p=0.75),
        A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
        A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
        # distortion
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.IAAPerspective(p=1),
            ],
            p=0.2,
        ),
        # noise transforms
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.IAASharpen(p=1),
                A.GaussianBlur(p=1),
            ],
            p=0.2,
        ),
    ]
    return A.Compose(augms, additional_targets={'osm': 'image'})(image=sample["image"], mask=sample["mask"],
                                                                 osm=sample["osm"])
