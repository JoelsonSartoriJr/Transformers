import torch.nn as nn
import torch
from .encoder_layer import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, 
                input_dim:int, 
                hid_dim:int, 
                n_layers:int, 
                n_heads:int, 
                pf_dim:int,
                dropout:float, 
                device:torch.device,
                max_length:int = 100)->None:
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderBlock(hid_dim, 
                                                n_heads, 
                                                pf_dim,
                                                dropout, 
                                                device) 
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src:list, src_mask:list)->list:
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src