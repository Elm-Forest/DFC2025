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
            scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.7
        ),

        # 随机选择 Resize 或 RandomCrop
        A.OneOf([
            A.Resize(height=size, width=size, p=1.0),  # 统一大小，不丢失信息
            A.RandomCrop(height=size, width=size, p=1.0)  # 随机裁剪，保留一定随机性
        ], p=1.0),  # 每次都会执行此操作，随机选择其中一个
        # 水平翻转
        A.HorizontalFlip(p=0.5),
        # 垂直翻转
        A.VerticalFlip(p=0.5),
        # 下采样
        A.Downscale(scale_range=(0.5, 0.75), p=0.05),

        # 保留变形和噪声相关增强
        A.OneOf(
            [
                # 弹性变换（Elastic Transform）
                # 通过模拟物体表面变形的方式进行图像的非线性变换。这种变换通常模拟了物体的弹性变形，在图像中产生扭曲的效果。
                # 这种变换常用于数据增强中，使得模型能够更好地处理物体表面不规则的形变，特别是目标在不同角度、不同位置时的表现。
                A.ElasticTransform(p=1),  # p=1表示一定会应用弹性变换

                # 光学畸变（Optical Distortion）
                # 模拟摄像机镜头的畸变效果。可以产生弯曲的视觉效果，使得图像中的直线部分弯曲。
                # 通常用于模拟摄像头或传感器的不完美，增强模型在实际应用中的鲁棒性。
                A.OpticalDistortion(p=1),  # p=1表示一定会应用光学畸变

                # 网格畸变（Grid Distortion）
                # 通过对图像应用网格状的局部变形，制造类似于透视畸变的效果。这种方法可以随机地变形图像，模拟不同角度的视角。
                # 它对模型有很好的帮助，特别是在目标物体存在畸变或视角变化的情况下。
                A.GridDistortion(p=1),  # p=1表示一定会应用网格畸变

                # 透视变换（Perspective Transform）
                # 通过改变图像的透视效果，模拟不同视角下的图像效果。例如，从一个高空或低角度观察物体，图像的透视关系会发生变化。
                # 透视变换让模型学习如何识别不同视角下的物体，增强模型的多视角识别能力。
                A.Perspective(p=1),  # p=1表示一定会应用透视变换
            ],
            p=0.1,  # p=0.1表示以10%的概率应用上述变换中的某一种
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
