import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 8000
BLOCK_SIZE = 128
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6   # IMPORTANT: must be 6

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.softmax(dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        return self.token_emb(x) + self.pos_emb(pos)

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = EmbeddingLayer(VOCAB_SIZE, EMBED_DIM, BLOCK_SIZE)

        self.blocks = nn.Sequential(
            *[TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)]
        )

        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

