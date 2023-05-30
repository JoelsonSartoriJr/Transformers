import torch.nn as nn
import torch
from multi_head_attention import MultiHeadAttentionLayer
from position import PositionwiseFeedforwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, 
                    hid_dim:int, 
                    n_heads:int, 
                    pf_dim:int, 
                    dropout:float, 
                    device:torch.device)->None:
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                        pf_dim, 
                                                                        dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg:list, enc_src:list, trg_mask:list, src_mask:list)->list:

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention