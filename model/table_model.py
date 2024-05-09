from torch import nn
from model.seq2mat import *

class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.seq2mat = Seq2Mat(config)
        self.layer   = nn.ModuleList([TableEncoding(config) for _ in range(2)])
        
    def forward(self, seq, mask):

        table = self.seq2mat(seq, seq)

        mask = mask.bool()
        td_mask = mask[:, None, :] & mask[:, :, None]

        mask  = td_mask

        if self.config.table_encoder == 'none':
            return table, mask

        for layer_module in self.layer:
            table = layer_module(table, mask)
        
        return table, mask

class TableEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.table_encoder == 'resnet':
            from model.resnet import ResNet
            self.encoder = ResNet(config)
        else:
            raise NotImplementedError

    def forward(self, table, mask):

        table = self.encoder(table, mask=mask)
        
        return table

