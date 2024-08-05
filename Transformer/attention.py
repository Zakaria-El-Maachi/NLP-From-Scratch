from torch import tensor, einsum
from torch.nn import Module, Sequential, Softmax, GELU, Linear, Dropout, LayerNorm
import math

class Attention(Module):
    def __init__(self, embedding_dim, num_heads, *args, **kwargs) -> None:
        assert embedding_dim % num_heads == 0, "The embedding dimension must be divisible by the number of heads"
        super().__init__(*args, **kwargs)
        self.embed_dim = embedding_dim
        self.n_head = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.queryLinear = Linear(self.head_dim, self.head_dim, bias=False)
        self.keyLinear = Linear(self.head_dim, self.head_dim, bias=False)
        self.valueLinear = Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc = Linear(self.head_dim, self.embed_dim)
        
        self.sft = Softmax(dim=-1)
        
    def forward(self, queries, keys, values, mask=None):
        # first reshape the tensors to be divided into heads
        qs, ks, vs = queries.shape, keys.shape, values.shape
        assert qs[0] == ks[0] and ks[0] == vs[0], "The batch size should be the same across all passed tensors"
        assert ks[1] == vs[1], "The sequence length should be the same across all the keys and the values"
        assert qs[2] == self.embed_dim and ks[2] == self.embed_dim and vs[2] == self.embed_dim, f"The embedding size should be equal to {self.embed_dim} across all tensors"
        
        queries = queries.reshape((qs[0], qs[1], self.n_head, self.head_dim))
        keys = keys.reshape((ks[0], ks[1], self.n_head, self.head_dim))
        values = values.reshape((vs[0], vs[1], self.n_head, self.head_dim))
        
        queries = self.queryLinear(queries)
        keys = self.keyLinear(keys)
        values = self.valueLinear(values)
        
        # b the batch size
        # q, k the lengths of the sequences
        # n the number of heads
        # h the head dim
        product = einsum("bqnh,bknh->bnqk", queries, keys)
        
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, qs[1], -1)
            product = product.masked_fill(expanded_mask == 0, -1e4)
        
        insight = self.sft(product / math.sqrt(self.embed_dim))
        
        # b the batch size
        # q, k, v the lengths of the sequences (k == v)
        # n the number of heads
        # h the head dim
        output = einsum("bnqk,bknh->bqnh", insight, values)
        
        return output.reshape((qs[0], qs[1], self.embed_dim))
    


class TransformerBlock(Module):
    def __init__(self, embedding_dim, num_heads, ffd_expansion = 4, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = Attention(embedding_dim, num_heads)
        self.norm1 = LayerNorm(embedding_dim)
        self.ffd = Sequential(
            Linear(embedding_dim, embedding_dim * ffd_expansion),
            GELU(),
            Linear(embedding_dim*ffd_expansion, embedding_dim)
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout = Dropout(dropout)
        
    
    def forward(self, input, mask=None):
        attention = self.attention(input, input, input, mask)
        output1 = self.dropout(self.norm1(attention + input))
        output2 = self.ffd(output1) + output1
        output = self.dropout(self.norm2(output2))
        return output