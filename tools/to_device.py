import torch


def ToDevice(src, device):
    if isinstance(src, torch.Tensor):
        return src.to(device)
    elif isinstance(src, (list, tuple)):
        return tuple(ToDevice(i, device) for i in src)
    else:
        return TypeError
