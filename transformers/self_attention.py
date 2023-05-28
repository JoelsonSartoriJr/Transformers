import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size:int, heads:int) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        
        assert (self.heads_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.heads_dim, embed_size)
        
    def forward(self, values:list, keys:list, queries:list, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        key_len = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = queries.reshape(N, query_len, self.heads, self.heads_dim)
        
        att = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        #queries shape: (N, query_len, heads, heads_dim)
        #keys shape: (N, key_len, heads, heads_dim)
        #att shape: (N, heads, query_len, key_len)
        
        if mask is None:
            attn = attn.masked_fill(mask==0, float("-1e20"))
        
        attention = torch.softmax(attn/(self.embed_size**(1/2)), dim=3)
        
        out = torch.eisum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.heads_dim
        )
        #attention shape: (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, heads_dim)
        #out shape: (N, query_len, heads, heads_dim)
        
        out = self.fc_out(out)
        return out