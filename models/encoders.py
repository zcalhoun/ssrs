import torch

from .base import resnet


def load(encoder_name):
    if encoder_name == "swav":
        return _load_swav()
    else:
        raise NotImplementedError


def _load_swav():
    """
    This model loads the weights from the SwAV model and places them
    onto this version of the ResNet model which allows the layers
    to be passed forward 

    """
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    state_dict = model.state_dict()

    # Remove keys that we don't need
    state_dict.pop("fc.bias")
    state_dict.pop("fc.weight")

    # Instantiate the version of ResNet that we want
    # and load the weights on top of this model.
    base_model = resnet.resnet50()
    base_model.load_state_dict(state_dict)
    return base_model
