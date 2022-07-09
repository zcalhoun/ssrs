import logging

import torch
from torchvision.models import resnet50 as r50

from .base import resnet


def load(encoder_name):
    if encoder_name == "swav":
        print("Loading swav-imagenet pretrained weights.")
        return _load_swav()
    elif encoder_name == "none":
        print("Loading encoder with no pretrained weights.")
        return _load_base()
    elif encoder_name == "imagenet":
        print("Loading supervised ResNet model.")
        return _load_imagenet()
    else:
        logging.error(f"Encoder {encoder_name} not implemented.")
        raise NotImplementedError

def _load_imagenet():
    """
    This model loads the weights from the SwAV model and places them
    onto this version of the ResNet model which allows the layers
    to be passed forward 

    """
    model = r50(pretrained=True)
    return _append_state_dict_to_resnet(model.state_dict())

def _load_base():
    # This only loads the base encoder model
    # with no pretrained weights
    base_model = resnet.resnet50(inter_features=True)
    return base_model

def _load_swav():
    """
    This model loads the weights from the SwAV model and places them
    onto this version of the ResNet model which allows the layers
    to be passed forward 

    """
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    return _append_state_dict_to_resnet(model.state_dict())

def _append_state_dict_to_resnet(state_dict):

    # Remove keys that we don't need
    state_dict.pop("fc.bias")
    state_dict.pop("fc.weight")

    # Instantiate the version of ResNet that we want
    # and load the weights on top of this model.
    # For semantic segmentation, we need inter_features
    # to be true.
    #
    # As we add more functionality, this piece of code
    # will need to change.
    base_model = resnet.resnet50(inter_features=True)
    base_model.load_state_dict(state_dict)
    return base_model