import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size:int,
        trg_vocab_size:int,
        src_pad_idx:int,
        trg_pad_idx:int,
        embed_size:int=256,
        num_layers:int=6,
        forward_expansion:int=4,
        heads:int=8,
        dropout:float=0,
        device:torch.device=torch.device('cpu'),
        max_length:int=100
    ):

        super(Transformer, self).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
    def make_src_mask(self, src:list)->list:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg:list)->list:
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src:list, trg:list):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encode_out = self.encoder(src, src_mask)
        out = self.decoder(trg, encode_out, src_mask, trg_mask)
        return out