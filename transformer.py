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


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # Define conv1d to mapping input to higher dim
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # Define conv2 to mapping input to original dim
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # norm layer
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, inputs): 
        #-------------------------dim-------------------------------- 
        # inputs [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        residual = inputs  # residual connection
        # Using Relu function after conv1
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) 
        #-------------------------dim-------------------------------- 
        # output [batch_size, d_ff, len_q]
        #----------------------------------------------------------------
        # Dimensionality reduction using conv2
        output = self.conv2(output).transpose(1, 2) 
        #-------------------------dim-------------------------------- 
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        # Residual connection with input and layer normalization
        output = self.layer_norm(output + residual) 
        #-------------------------dim-------------------------------- 
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        return output

# A function that generates a sinusoidal position encoding table for introducing position information into the Transformer
def get_sin_enc_table(n_position, embedding_dim):
    #-------------------------dim--------------------------------
    # n_position: maximum length of input sequence
    # embedding_dim: dimension of word embedding vector
    #-----------------------------------------------------------------    
    # Initialize the sinusoidal position encoding table based on the position and dimension information
    sinusoid_table = np.zeros((n_position, embedding_dim))    
    # Traverse all positions and dimensions and calculate the angle value
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle    
    # Calculate sine and cosine values
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维
    #-------------------------dim--------------------------------
    # sinusoid_table shape: [n_position, embedding_dim]
    #----------------------------------------------------------------   
    return torch.FloatTensor(sinusoid_table)

#Define padding attention mask function
def get_attn_pad_mask(seq_q, seq_k):
    #-------------------------dim--------------------------------
    # seq_q dim [batch_size, len_q]
    # seq_k dim [batch_size, len_k]
    #-----------------------------------------------------------------
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # generate tensor of boolean
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # <PAD>token encoding number is 0
    #-------------------------dim--------------------------------
    # pad_attn_mask dim [batch_size,1,len_k]
    #-----------------------------------------------------------------
    # Reshape into a tensor of the same shape as the attention score
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    #-------------------------dim--------------------------------
    # pad_attn_mask dim [batch_size,len_q,len_k]
    #-----------------------------------------------------------------
    return pad_attn_mask 


