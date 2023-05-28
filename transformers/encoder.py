import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size:int,
        embed_size:int,
        num_layers:int,
        heads:int,
        device:torch.device,
        forward_expansion:int,
        dropout:float,
        max_length:int
    ) -> None:
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:list, mask)->list:
        N, seq_length = x.shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(pos))
        )
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out