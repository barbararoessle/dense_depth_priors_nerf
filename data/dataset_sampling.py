import torch
from torch.utils.data import random_split

def compute_samples_per_subset(sample_count, validate_on_at_least_n_samples):
    validate_on_at_least_n_samples = min(validate_on_at_least_n_samples, sample_count)
    number_subsets = int(sample_count / validate_on_at_least_n_samples)
    samples_per_subset = int(sample_count / number_subsets)
    extra_sample_subsets = sample_count % samples_per_subset
    normal_subsets = number_subsets - extra_sample_subsets
    return samples_per_subset, normal_subsets, extra_sample_subsets

def create_random_subsets(dataset, validate_on_at_least_n_samples, device='cpu'):
    samples_per_subset, normal_subsets, extra_sample_subsets = compute_samples_per_subset(len(dataset), validate_on_at_least_n_samples)
    subsets = random_split(dataset, (samples_per_subset,) * normal_subsets + (samples_per_subset + 1,) * extra_sample_subsets, \
        torch.Generator(device=device))
    return subsets
