import argparse
import os

import segmentation_models_pytorch as smp
import torch

from source.mit_unet.network_mit_unet import Net


def creatModel(args,
               in_channels=1,
               activation=None,
               encoder_weights="imagenet",
               encoder_name='mit_b4'):
    encoder_name = args.encoder_name
    if args.model_name == 'sam2':
        print('Building SAM2+Segformer')
        model = smp.Segformer(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name='tu-sam2_hiera_large',
            img_size=args.crop_size,
        )
    elif args.model_name == 'mit_unet':
        print('Building mit_unet')
        model = Net(phi=args.model_size,
                    pretrained=args.pretrained,
                    num_classes=len(args.classes) + 1)
    elif args.model_name == 'segformer':
        print('Building Segformer')
        model = smp.Segformer(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'deeplab':
        print('Building DeepLabV3+')
        model = smp.DeepLabV3Plus(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'uper':
        print('Building UPerNet')
        model = smp.UPerNet(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'unetp':
        print('Building Unet++')
        model = smp.UnetPlusPlus(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            decoder_attention_type='scse',
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'psp':
        print('Building PSPNet')
        model = smp.PSPNet(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'ma':
        print('Building MAnet')
        model = smp.MAnet(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    elif args.model_name == 'link':
        print('Building Linknet')
        model = smp.Linknet(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    else:
        print('Building Segformer')
        model = smp.Segformer(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=args.crop_size,
        )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--model_name', default="deeplab")
    parser.add_argument('--encoder_name', default="tu-convnextv2_huge")
    parser.add_argument('--save_model', default="../model")
    args = parser.parse_args()
    model = creatModel(args,
                       in_channels=1,
                       activation=None,
                       encoder_weights="imagenet",
                       encoder_name='mit_b4')
    torch.save(model.state_dict(), os.path.join(args.save_model, f"model_swinL_size_test.pth"))
