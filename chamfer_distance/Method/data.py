import torch


def to_valid_tensor(source_tensor):
    valid_tensor = torch.nan_to_num(source_tensor, 0.0)
    return valid_tensor
