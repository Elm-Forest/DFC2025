import segmentation_models_pytorch as smp


def creatModel(args,
               in_channels=1,
               activation=None,
               encoder_weights="imagenet",
               encoder_name='mit_b4'):
    model = smp.Segformer(
        classes=len(args.classes) + 1,
        in_channels=in_channels,
        activation=activation,
        encoder_weights=encoder_weights,
        encoder_name=encoder_name,
    )
    return model
