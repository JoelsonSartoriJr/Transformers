import torch
import torch.nn as nn
from decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size: int,
            embed_size:int,
            num_layers:int,
            heads:int,
            forward_expansion:int,
            dropout:float,
            device:torch.device,
            max_length:int
        ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:list, encode_out:list, src_mask:int, trg_mask:int)->list:
        N, seq_length = x.shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(pos))
        
        for layer in self.layers:
            x = layer(x, encode_out, encode_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out