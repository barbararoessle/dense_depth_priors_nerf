import torch

def add_missing_depth(depth, valid_depth, p=0.1, invalid_depth_value=0):
    n_pixels = valid_depth.numel()
    n_valid = valid_depth.sum()
    p_before = float(n_pixels - n_valid) / float(n_pixels)
    p_gap = p - p_before
    if p_gap <= 0.:
        return depth, valid_depth
    else:
        p_to_be_invalidated = p_gap * float(n_pixels) / float(n_valid)
    invalid = torch.rand_like(depth) < p_to_be_invalidated
    valid_depth[invalid] = False
    depth[invalid] = invalid_depth_value
    return depth, valid_depth

def add_quadratic_depth_noise(depth, valid_depth, a=1.68e-3, b=6.58e-3, c=4.78e-2):
    std = a * depth[valid_depth].pow(2) + b * depth[valid_depth] + c
    noise = torch.randn_like(std) * std
    depth[valid_depth] = (depth[valid_depth] + noise).clamp(min=0.)
    return depth
