import torch
from torch import nn

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask, dim=1):
        x_shape = list(x.shape)
        mask_shape = list(attention_mask.shape)
        for i in range(1,len(x_shape)):
            if i == dim:
                x_shape[i] = mask_shape[-1]
            else:
                x_shape[i] = 1
        mask_reshaped = attention_mask.reshape(x_shape)
        x_pooled, _ = torch.max(x.masked_fill(~mask_reshaped.bool(),-1e4), dim=dim)
        return x_pooled