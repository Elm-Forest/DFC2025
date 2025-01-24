import torch
import torch.nn as nn

from .segformer import SegFormer


class Net(nn.Module):
    def __init__(self, phi="b0", pretrained="pretrained/mit_b0.pth", num_classes=1):
        super(Net, self).__init__()
        self.segFormer = SegFormer(phi=phi, num_classes=num_classes)
        self.pretrained = pretrained
        self.name = "MiT-Unet"
        if pretrained is not None:
            self.load_pretrained_model(self.segFormer)

    def load_pretrained_model(self, model):
        pretrained_weight = self.pretrained
        state_dict = model.state_dict()
        model_dict = {}
        load_key, no_load_key = [], []
        pretrain_dict = torch.load(pretrained_weight, map_location=f"cuda:0")
        # pretrain_dict = {k.replace('module.', ''): v for k, v in pretrain_dict.items()}
        pretrain_dict_items = pretrain_dict.items() if "state_dict" not in pretrain_dict else pretrain_dict[
            "state_dict"].items()
        for k, v in pretrain_dict_items:
            k = "backbone." + k
            if k in state_dict and v.shape == state_dict[k].shape:
                model_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        state_dict.update(model_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loading pretrained weight: '{pretrained_weight}' done.")
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    def forward(self, x):
        logist = self.segFormer(x)[0]
        return logist
