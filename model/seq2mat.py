import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.activations import ACT2FN


class Seq2Mat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduce = nn.Linear(768*3, 768)
        self.norm   = T5LayerNorm(768, 1e-06)
        self.activation = ACT2FN['gelu']
        

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        seq = x.clone()
        batch_size = seq.shape[0]
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        max_len = seq.shape[1]
        maxp = seq.transpose(1,2)

        # calculate context embedding in table encoder
        z = torch.ones_like(x).to('cuda')
        for i in range(max_len):
            diag = x.diagonal(dim1=1,dim2=2,offset=-i)
            maxp = torch.max(maxp[:,:,:max_len-i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len-i)] for b in range(batch_size)] 
            lineyup = [[j+i for j in range(max_len-i)] for b in range(batch_size)] 
            linexdown = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len-i)] for b in range(batch_size)]   
            z[bb,linexup,lineyup,:] = maxp.permute(0,2,1)
            z[bb,linexdown,lineydown,:] = maxp.permute(0,2,1)
        
        t = torch.cat([x, y, z], dim=-1)
        t = self.reduce(t)
        t = self.activation(t)

        return t