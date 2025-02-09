import os

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset as BaseDataset

from source import transforms as T


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def load_png(path):
    src = Image.open(path, "r")
    src = np.array(src)
    return src.astype(np.uint8)


def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform


def save_img(path, img, crs, transform):
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=img.shape[0],
            dtype=img.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()


class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False):
        self.fns = label_list
        self.augm = T.train_augm if train else T.valid_augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_grayscale(self.fns[idx].replace("labels", "sar_images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            # data = self.augm({"image": img, "mask": msk})
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)


class Dataset_limit(BaseDataset):
    def __init__(self, label_list, classes=None, size=1024, train=False,
                 use_binary_aug=False, cls_id=4, fiter_threshold=0.01):
        self.augm = T.train_augm if train else T.valid_augm
        self.augm = T.train_augm_binary if train and use_binary_aug else self.augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale
        self.cls_id = cls_id
        self.fiter_threshold = fiter_threshold
        # 过滤 label_list，仅保留 cls_id 面积占比 > self.cls_id 的样本
        self.fns = self.filter_labels(label_list) if fiter_threshold > 0 else label_list

    def filter_labels(self, label_list):
        """ 过滤 4 的占比 > 0.01 的 mask 文件 """
        valid_labels = []
        for label_path in label_list:
            msk = self.load_grayscale(label_path)  # 使用 load_grayscale 读取 tif
            msk_array = np.array(msk)  # 转换为 numpy 数组

            total_pixels = msk_array.size  # 总像素数
            num_pixels = np.sum(msk_array == self.cls_id)  # 计算像素数量
            proportion = num_pixels / total_pixels  # 计算占比

            if proportion > self.fiter_threshold:
                valid_labels.append(label_path)

        print(f"过滤后剩余 {len(valid_labels)} 个样本（{self.cls_id} 的面积占比 > {self.fiter_threshold}）")
        return valid_labels

    def __getitem__(self, idx):
        img = self.load_grayscale(self.fns[idx].replace("labels", "sar_images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)


def linear_mapping(mask, new_min=1, new_max=255):
    # 将0到8的值线性映射到0到255之间
    old_min, old_max = mask.min(), mask.max()
    return ((mask - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


class Dataset_Fusion(BaseDataset):
    def __init__(self, label_list, ensemble_folders, classes=None, size=1024, train=False):
        self.fns = label_list
        self.ensemble_folders = ensemble_folders
        self.augm = T.train_augm_multiple if train else T.valid_augm_multiple
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_grayscale = load_grayscale
        self.load_png = load_png

    def __getitem__(self, idx):
        original_path = self.fns[idx]
        img = self.load_grayscale(original_path.replace("labels", "sar_images"))
        file_name_no_extension = os.path.splitext(os.path.basename(original_path))[0]

        ensemble_outputs = []
        for folder in self.ensemble_folders:
            model_output_path = os.path.join(folder, f"{file_name_no_extension}.tif")
            im_array = self.load_png(model_output_path)
            im_array = linear_mapping(im_array)
            ensemble_outputs.append(np.expand_dims(im_array, axis=-1))
        ensemble_outputs = np.concatenate(ensemble_outputs, axis=-1)
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk, "image_ensemble": ensemble_outputs}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk, "image_ensemble": ensemble_outputs}, self.size)

        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "img_esb": data["image_ensemble"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)
