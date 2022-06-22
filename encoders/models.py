import torch


def load_encoder(encoder_name):
    if encoder_name == "swav":
        return SwavEncoder()
    else:
        raise NotImplementedError


def _load_swav():
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    model.fc = torch.nn.Identity()
    return model
