from base import unet


def load(decoder_name):
    if decoder_name == "unet":
        return _load_unet()

    else:
        raise NotImplementedError


def _load_unet():
    raise NotImplementedError

    model = unet.UNet(
        n_class=args["dataset"]["class_num"],
        encoder_name=args["encoder_name"],
        pretrained=eval(args["imagenet"]),
        aux_loss=aux_loss,
        use_emau=args["use_emau"],
        use_ocr=args["use_ocr"],
    )

    return model

