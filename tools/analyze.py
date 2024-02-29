import thop
import torch
import torch.nn as nn


def count_parameters(model: nn.Module):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def profile_model(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    return {"FLOPs": flops / 1e9 * 2, "params": params}  # unit: one billion, backward and forward(x2)
