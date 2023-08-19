import torch
import torch.nn as nn


@torch.no_grad()
def infer_fc_input_dim(module, input_shape):
    dummy_input = torch.rand(2, *input_shape)
    dummy_output = module(dummy_input)
    infer_dim = torch.flatten(dummy_output, start_dim=1).shape[-1]
    return infer_dim


def xavier_uniform_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def orthogonal_init(m, scale=1.0):
    if isinstance(m, nn.Linear):
        lasagne_orthogonal(m.weight, scale)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def lasagne_orthogonal(tensor, scale=1.0):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)
    u, _, v = torch.svd(flattened)
    q = u if u.shape == flattened.shape else v  # pick the one with the correct shape
    q = q.reshape(tensor.shape)
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(scale)
    return tensor


def clamp(input, min, max):
    # Currently torch.clamp() does not support tensor arguments for min/max
    # while torch.min() / max() does not support float arguments
    if isinstance(min, torch.Tensor) and isinstance(max, torch.Tensor):
        clipped = torch.max(torch.min(input, max), min)
    else:
        clipped = torch.clamp(input, min=min, max=max)
    return clipped
