import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import Optional
import numpy as np

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, seq_len, drop_prob=0):
        super(TransformerEmbedding, self).__init__()

        self.encoding = torch.zeros(seq_len, d_model)
        self.encoding.requires_grad = False 

        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()
  
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.dropEmbed = nn.Dropout(drop_prob)
        
    def forward(self, x):
        batch_size ,seq_len = x.shape
        return self.dropEmbed(self.embedding(x) + self.encoding[:seq_len, :])

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=False, mask=False):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.softmax_layer = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.q_layer.weight)
        nn.init.xavier_uniform_(self.k_layer.weight)
        nn.init.xavier_uniform_(self.v_layer.weight)

        self.mask = mask

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self,q , k , v):
        batch_size, seq_len = q.size(0), q.size(1)
        
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attention = self.softmax_layer(torch.matmul(q, k.transpose(-2,-1))/(q.size(3) ** 0.5))

        if self.mask:
            mask_ = torch.triu(torch.ones_like(attention), diagonal=1)
            attention = attention.masked_fill(mask_ == 1, float('-inf'))
        
        activation = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size,  attention.size(2), -1)

        return activation

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4, qk_norm=False, dropout=0):
        super(TransformerEncoder,self).__init__()
        self.attention = MultiHeadSelfAttention(d_model,num_heads,qk_norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio*d_model, d_model)
        )

    def forward(self,x):
        skip_1 = x.clone()
        x_1 = self.attention(x, x, x)
        
        x_1 = self.dropout1(x_1)
        output_1 = self.norm1(x_1 + skip_1)

        skip_2 = output_1.clone()
        x_2 = self.mlp(output_1)
        
        x_2 = self.dropout2(x_2)
        output_2 = self.norm2(x_2 + skip_2)

        return output_2

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4, qk_norm=False, dropout=0):
        super(TransformerDecoder, self).__init__()
        self.masked_attention = MultiHeadSelfAttention(d_model, num_heads, qk_norm, mask=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, qk_norm)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model)
        )
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, q, k):
        skip_1 = x.clone()
        x_1 = self.masked_attention(x, x, x)
        x_1 = self.dropout_1(x_1)
        output_1 = self.norm_1(x_1 + skip_1)

        skip_2 = output_1.clone()
        x_2 = self.attention(q, k, output_1)
        x_2 = self.dropout_2(x_2)
        output_2 = self.norm_2(x_2 + skip_2)

        skip_3 = output_2.clone()
        x_3 = self.mlp(output_2)
        x_3 = self.dropout_3(x_3)
        output_3 = self.norm_3(x_3 + skip_3)

        return output_3
