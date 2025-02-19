import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
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
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
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

# Define Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()        
        self.enc_self_attn = MultiHeadAttention()  # Multi-head self-attention layer       
        self.pos_ffn = PoswiseFeedForwardNet()     # Position-wise feed-forward network

    def forward(self, enc_inputs, enc_self_attn_mask):
        #-------------------------dim--------------------------------
        # enc_inputs dim: [batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask dim: [batch_size, seq_len, seq_len]
        #-----------------------------------------------------------------
        # Input the same Q, K, V into the multi-head self-attention layer, 
        # and the returned attn_weights will have an additional head dimension.
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, enc_self_attn_mask)
        #-------------------------dim--------------------------------
        # enc_outputs dim: [batch_size, seq_len, embedding_dim] 
        # attn_weights dim: [batch_size, n_heads, seq_len, seq_len]      
        # Input the outputs of the multi-head self-attention into the position-wise feed-forward network layer
        enc_outputs = self.pos_ffn(enc_outputs)  # The dimension is the same as enc_inputs
        #-------------------------dim--------------------------------
        # enc_outputs dim: [batch_size, seq_len, embedding_dim] 
        #-----------------------------------------------------------------
        return enc_outputs, attn_weights  # Return the encoder outputs and attention weights of each encoder layer

n_layers = 6  # encoder layer number
class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()        
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding) # embedding layer
        self.pos_emb = nn.Embedding.from_pretrained( \
          get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True) # position embedding layer
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))# encoder layers

    def forward(self, enc_inputs):  
        #-------------------------dim--------------------------------
        # enc_inputs dim: [batch_size, source_len]
        #-----------------------------------------------------------------
        # Create position index sequence from 1 to source_len
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        #-------------------------dim--------------------------------
        # pos_indices dim: [1, source_len]
        #-----------------------------------------------------------------
        # add word embedding and pos embedding to input [batch_size, source_len, embedding_dim]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        #-------------------------dim--------------------------------
        # enc_outputs dim: [batch_size, seq_len, embedding_dim]
        #-----------------------------------------------------------------
        # generate self attention padding mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        #-------------------------dim--------------------------------
        # enc_self_attn_mask dim: [batch_size, len_q, len_k]        
        #-----------------------------------------------------------------
        enc_self_attn_weights = [] # init enc_self_attn_weights
        # through encoding layer [batch_size, seq_len, embedding_dim]
        for layer in self.layers: 
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        #-------------------------dim--------------------------------
        # enc_outputs dim: [batch_size, seq_len, embedding_dim] dim is the same as enc_inputs
        # enc_self_attn_weights is a list，each element dim:[batch_size, n_heads, seq_len, seq_len]          
        #-----------------------------------------------------------------
        return enc_outputs, enc_self_attn_weights # return encoder output and attention weights

# A function that generates subsequent attention mask to ignore future information in multi-head self-attention calcluations
def get_attn_subsequent_mask(seq):
    #-------------------------dim--------------------------------
    # seq dim: [batch_size, seq_len(Q)=seq_len(K)]
    #-----------------------------------------------------------------
    # get the shape of input
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  
    #-------------------------dim--------------------------------
    # attn_shape is a one dim tensor [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # Creating an upper triangular matrix using numpy(triu = triangle upper)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    #-------------------------dim--------------------------------
    # subsequent_mask dim: [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # Convert numpy array to PyTorch tensor and set data type to byte (boolean)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    #-------------------------dim--------------------------------
    # Returned subsequent_mask dim: [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    return subsequent_mask

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()        
        self.dec_self_attn = MultiHeadAttention() # multi-head self-attention layer     
        self.dec_enc_attn = MultiHeadAttention()  # multi-head self-attention layer, connect encoder and decoder       
        self.pos_ffn = PoswiseFeedForwardNet() # position wise feed forward network

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        #-------------------------dim--------------------------------
        # dec_inputs dim: [batch_size, target_len, embedding_dim]
        # enc_outputs dim: [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask dim: [batch_size, target_len, target_len]
        # dec_enc_attn_mask dim: [batch_size, target_len, source_len]
        #-----------------------------------------------------------------      
        # Input the same Q, K, V into the multi-head self-attention layer
        # use the same Q,K,V as input
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                        dec_inputs, dec_self_attn_mask)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        # dec_self_attn dim: [batch_size, n_heads, target_len, target_len]
        #-----------------------------------------------------------------        
        # Feed the decoder output and encoder output into the multi-head self-attention layer
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                      enc_outputs, dec_enc_attn_mask)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        # dec_enc_attn dim: [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # Input position-wise feed-forward network layer
        dec_outputs = self.pos_ffn(dec_outputs)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        # dec_self_attn dim: [batch_size, n_heads, target_len, target_len]
        # dec_enc_attn dim: [batch_size, n_heads, target_len, source_len]   
        #-----------------------------------------------------------------
        # Returns the decoder layer output, self-attention and decoder-encoder attention weights for each layer
        return dec_outputs, dec_self_attn, dec_enc_attn

n_layers = 6  # Set Decoder layer number
class Decoder(nn.Module):
    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding) # word embedding layer
        self.pos_emb = nn.Embedding.from_pretrained( \
           get_sin_enc_table(corpus.tgt_len+1, d_embedding), freeze=True) # position embedding layer       
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]) # Stacking multiple decoding layers

    def forward(self, dec_inputs, enc_inputs, enc_outputs): 
        #-------------------------dim--------------------------------
        # dec_inputs dim: [batch_size, target_len]
        # enc_inputs dim: [batch_size, source_len]
        # enc_outputs dim: [batch_size, source_len, embedding_dim]
        #-----------------------------------------------------------------   
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        #-------------------------dim--------------------------------
        # pos_indices dim: [1, target_len]
        #-----------------------------------------------------------------
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        #-----------------------------------------------------------------
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # padding mask
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) # Subsequent Mask
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask \
                                       + dec_self_attn_subsequent_mask), 0) 
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # decoder-encoder mask
        #-------------------------dim--------------------------------        
        # dec_self_attn_pad_mask dim: [batch_size, target_len, target_len]
        # dec_self_attn_subsequent_mask dim: [batch_size, target_len, target_len]
        # dec_self_attn_mask dim: [batch_size, target_len, target_len]
        # dec_enc_attn_mask dim: [batch_size, target_len, source_len]
         #-----------------------------------------------------------------       
        dec_self_attns, dec_enc_attns = [], [] # Init dec_self_attns, dec_enc_attns
        # decoder layers [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, 
                                               dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        # dec_self_attns is a list，each element dim: [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns is a list，each element dim: [batch_size, n_heads, target_len, source_len]
        #----------------------------------------------------------------- 
              
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()        
        self.encoder = Encoder(corpus) 
        self.decoder = Decoder(corpus) 
        # Define a linear projection layer to transform the decoder output into a probability distribution of the target vocabulary size
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab), bias=False)
    def forward(self, enc_inputs, dec_inputs):
        #-------------------------dim--------------------------------
        # enc_inputs dim: [batch_size, source_seq_len]
        # dec_inputs dim: [batch_size, target_seq_len]
        #-----------------------------------------------------------------        
        # Call encoder   
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        #-------------------------dim--------------------------------
        # enc_outputs dim: [batch_size, source_len, embedding_dim]
        # enc_self_attns is a list, each element dim: [batch_size, n_heads, src_seq_len, src_seq_len]        
        #-----------------------------------------------------------------
        # Call decoder   
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #-------------------------dim--------------------------------
        # dec_outputs dim: [batch_size, target_len, embedding_dim]
        # dec_self_attns is a list, each element dim: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
        # dec_enc_attns is a list, each element dim: [batch_size, n_heads, tgt_seq_len, src_seq_len]   
        #-----------------------------------------------------------------
        # call projection
        dec_logits = self.projection(dec_outputs)  
        #-------------------------dim--------------------------------
        # dec_logits dim: [batch_size, tgt_seq_len, tgt_vocab_size]
        #-----------------------------------------------------------------
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


sentences = [
    ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
    ['我 爱 学习 人工智能', 'I love studying AI'],
    ['深度学习 改变 世界', ' DL changed the world'],
    ['自然语言处理 很 强大', 'NLP is powerful'],
    ['神经网络 非常 复杂', 'Neural-networks are complex'] ]

from collections import Counter # 导入Counter 类
class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        # Calculate the maximum sentence length for the source and target languages, and add 1 and 2 respectively to accommodate fillers and special symbols
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2
        # 创建源语言和目标语言的词汇表
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        # 创建索引到单词的映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
    # 定义创建词汇表的函数
    def create_vocabularies(self):
        # 统计源语言和目标语言的单词频率
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())        
        # 创建源语言和目标语言的词汇表，并为每个单词分配一个唯一的索引
        src_vocab = {'<pad>': 0, **{word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 
                     **{word: i+3 for i, word in enumerate(tgt_counter)}}        
        return src_vocab, tgt_vocab
    # 定义创建批次数据的函数
    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []
        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]
            # 将源语言和目标语言的句子转换为索引序列
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] \
                         for word in tgt_sentence.split()] + [self.tgt_vocab['<eos>']]            
            # 对源语言和目标语言的序列进行填充
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))            
            # 将处理好的序列添加到批次中
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']] * \
                                    (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])        
          # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)            
        return input_batch, output_batch, target_batch

# 创建语料库类实例
corpus = TranslationCorpus(sentences)

model = Transformer(corpus) # 创建模型实例
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器
epochs = 100 # 训练轮次
t0 = time.time()

for epoch in range(epochs): # 训练100轮
    optimizer.zero_grad() # 梯度清零
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size) # 创建训练数据
    outputs, _, _, _ = model(enc_inputs, dec_inputs) # 获取模型输出 
    loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1)) # 计算损失
    if (epoch + 1) % 20 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward()# 反向传播        
    optimizer.step()# 更新参数

t1 = time.time()
print('time_cost:', t1 - t0)
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 定义贪婪解码器函数
def greedy_decoder(model, enc_input, start_symbol):
    # 对输入数据进行编码，并获得编码器输出及自注意力权重
    enc_outputs, enc_self_attns = model.encoder(enc_input)    
    # 初始化解码器输入为全零张量，大小为 (1, 5)，数据类型与 enc_input 一致
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)    
    # 设置下一个要解码的符号为开始符号
    next_symbol = start_symbol    
    # 循环5次，为解码器输入中的每一个位置填充一个符号
    for i in range(0, 5):
        # 将下一个符号放入解码器输入的当前位置
        dec_input[0][i] = next_symbol        
        # 运行解码器，获得解码器输出、解码器自注意力权重和编码器-解码器注意力权重
        dec_output, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # 将解码器输出投影到目标词汇空间
        projected = model.projection(dec_output)        
        # 找到具有最高概率的下一个单词
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]        
        # 将找到的下一个单词作为新的符号
        next_symbol = next_word.item()  
    # 返回解码器输入，它包含了生成的符号序列
    dec_outputs = dec_input
    return dec_outputs

# enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1,test_batch=True) 
# predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs) # 用模型进行翻译
# predict = predict.view(-1, len(corpus.tgt_vocab)) # 将预测结果维度重塑
# predict = predict.data.max(1, keepdim=True)[1] # 找到每个位置概率最大的单词的索引
# # 解码预测的输出，将所预测的目标句子中的索引转换为单词
# translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
# # 将输入的源语言句子中的索引转换为单词
# input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
# print(input_sentence, '->', translated_sentence) # 打印原始句子和翻译后的句子


epochs = 1 
for epoch in range(epochs):
    # 用贪婪解码器生成翻译文本
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True) 
    # 使用贪婪解码器生成解码器输出
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=corpus.tgt_vocab['<sos>'])
    # 将解码器输入转换为单词序列
    greedy_dec_output_words = [corpus.tgt_idx2word[n.item()] for n in greedy_dec_input.squeeze()]
    # 打印编码器输入和贪婪解码器生成的文本
    enc_inputs_words = [corpus.src_idx2word[code.item()] for code in enc_inputs[0]]
    print(enc_inputs_words, '->', greedy_dec_output_words)
