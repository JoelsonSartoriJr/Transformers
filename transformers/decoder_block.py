import torch
import torch.nn as nn
from self_attention import SelfAttention
from transformer_block import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size:int, heads:int, forward_expansion:int, dropout:float, device:torch.device) -> None:
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:list, value:int, key:int, src_mask:int, trg_mask:int)->list:
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out