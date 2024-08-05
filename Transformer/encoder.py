import torch
from torch.nn import Module, Embedding, Dropout, ModuleList, Linear
from attention import TransformerBlock


def positionalEncoding(seq_len, embed_dim, n = 10000):
    assert embed_dim%2 == 0, "The embedding dimension must be even"
    
    pos = seq_len
    d = embed_dim
    
    positions = torch.arange(0, pos).unsqueeze(1)
    powers = torch.pow(n, torch.arange(0, d//2)/d)
    embeddings = torch.zeros((pos, d))
    
    embed_in = positions / powers
    
    embeddings[:, 0::2] = torch.sin(embed_in)
    embeddings[:, 1::2] = torch.cos(embed_in)
    
    return embeddings



class Encoder(Module):
    def __init__(self, vocab_size, embedding_dim, n_layers, n_heads, max_length, ffd = 4, dropout = 0.2, device = 'cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positionalEncoding = positionalEncoding(max_length, embedding_dim)
        self.layers = ModuleList(
            [TransformerBlock(embedding_dim, n_heads, ffd, dropout) for _ in range(n_layers)]
        )
        self.dropout = Dropout(dropout)
        self.head = Linear(embedding_dim, vocab_size)
        
    def forward(self, x, mask=None):
        output = self.dropout(self.embedding(x) + self.positionalEncoding.unsqueeze(dim=0).expand(x.shape[0], self.max_length, self.embedding_dim).to(x.device))
        
        for layer in self.layers:
            output = layer(output, mask)
            
        return self.head(output)
        