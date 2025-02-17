import numpy as np
import torch
import torch.nn as nn
d_k = 64
d_v = 64 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()        
    
    def forward(self, Q, K, V, attn_mask):
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        
        # scores [batch_size, n_heads, len_q, len_k]        
        # attn_mask [batch_size, n_heads, len_q, len_k]
        
        scores.masked_fill_(attn_mask, -1e9) 
        weights = nn.Softmax(dim=-1)(scores) 
        # weights [batch_size, n_heads, len_q, len_k]
        
        context = torch.matmul(weights, V) 
        # context [batch_size, n_heads, len_q, dim_v]
        
        return context, weights

d_embedding = 512  # Embedding dim
n_heads = 8  # Multi-Head Attention head number
batch_size = 3 
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, Q, K, V, attn_mask): 
        #-------------------------dim-------------------------------- 
        # Q K V [batch_size, len_q/k/v, embedding_dim] 
        #-----------------------------------------------------------------        
        residual, batch_size = Q, Q.size(0) 

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        #-------------------------dim-------------------------------- 
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        #----------------------------------------------------------------- 
        # copy attn_mask to multi-head attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        #-------------------------dim-------------------------------- 
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #----------------------------------------------------------------- 
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #-------------------------dim-------------------------------- 
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        #----------------------------------------------------------------- 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        #-------------------------dim-------------------------------- 
        # context [batch_size, len_q, n_heads * dim_v]
        #-----------------------------------------------------------------        
        output = self.linear(context) 
        #-------------------------dim-------------------------------- 
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------        
        output = self.layer_norm(output + residual)
        #-------------------------dim-------------------------------- 
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------        
        return output, weights