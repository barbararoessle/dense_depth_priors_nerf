import torch

def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())