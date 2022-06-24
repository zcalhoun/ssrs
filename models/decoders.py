import logging
from .base import unet


def load(decoder_name, encoder):
    if decoder_name == "unet":
        return _load_unet(encoder)

    else:
        raise NotImplementedError


def _load_unet(encoder):

    # Arguments taken from
    # https://github.com/bohaohuang/mrs/blob/master/network/unet.py
    # lines 156-159
    # This assumes that the encoder has chans
    try:
        in_chans = encoder.chans[:-1]
        out_chans = encoder.chans[1:]
        margins = [0, 0, 0, 0]
    except AttributeError:
        logging.error("Channels is not defined for encoder.")
        raise AttributeError("Encoder must have channels defined.")

    model = unet.UnetDecoder(in_chans, out_chans, margins, 1)

    return model

