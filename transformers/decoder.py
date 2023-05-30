import torch.nn as nn
import torch
from .decoder_layer import DecoderLayer
class Decoder(nn.Module):
    def __init__(self, 
                output_dim:int, 
                hid_dim:int, 
                n_layers:int, 
                n_heads:int, 
                pf_dim:int, 
                dropout:float, 
                device:torch.device,
                max_length:int = 100)->None:
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                    n_heads, 
                                                    pf_dim, 
                                                    dropout, 
                                                    device)
                                    for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg:list, enc_src:list, trg_mask:list, src_mask:list)->list:

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention