from multi_head_attention import MultiHeadAttentionLayer
from position import PositionwiseFeedforwardLayer
import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self,hid_dim:int, n_heads:int, pf_dim:int, dropout:float, device: torch.device) -> None:
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                        pf_dim, 
                                                                        dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src:list, src_mask:list)->list:

        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src