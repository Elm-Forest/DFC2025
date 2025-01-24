import argparse

import segmentation_models_pytorch as smp

from source.mit_unet.network_mit_unet import Net


def creatModel(args,
               in_channels=1,
               activation=None,
               encoder_weights="imagenet",
               encoder_name='mit_b4'):
    # model = smp.DeepLabV3Plus(
    #     classes=len(args.classes) + 1,
    #     in_channels=in_channels,
    #     activation=activation,
    #     encoder_weights=encoder_weights,
    #     encoder_name=encoder_name,
    # )
    if args.model_name == 'sam2':
        model = smp.Segformer(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name='tu-sam2_hiera_large',
        )
    elif args.model_name == 'mit_unet':
        model = Net(phi='b1', pretrained=args.pretrained)
    else:
        model = smp.Segformer(
            classes=len(args.classes) + 1,
            in_channels=in_channels,
            activation=activation,
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
        )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--model_name', default="sam2")
    parser.add_argument('--save_model', default="../model")
    args = parser.parse_args()
    model = creatModel(args,
                       in_channels=1,
                       activation=None,
                       encoder_weights="imagenet",
                       encoder_name='mit_b4')
    torch.save(model.state_dict(), os.path.join(args.save_model, f"model_swinL_size_test.pth"))
