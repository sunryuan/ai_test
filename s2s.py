# Build a corpus where each line contains three sentences: Chinese, English (decoder input), and the target English translation
import torch.nn as nn  # Import torch.nn library
import numpy as np  # Import numpy
import torch  # Import torch
import random  # Import random library
import matplotlib.pyplot as plt
import seaborn as sns

sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']
]

word_list_cn, word_list_en = [], []  # Initialize Chinese and English vocabularies
# Traverse each sentence and add words to the vocabulary
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())


max_dec_len = 6

# Deduplicate to get vocabularies without duplicate words
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))
# if '<pad>' not in word_list_en:
#     word_list_en.insert(0,'<pad>')
# Build word-to-index mapping
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}
# Build index-to-word mapping
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}
# Calculate the size of the vocabularies
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)
print("Number of sentences:", len(sentences))  # Print the number of sentences
print("Size of Chinese vocabulary:", voc_size_cn)  # Print the size of the Chinese vocabulary
print("Size of English vocabulary:", voc_size_en)  # Print the size of the English vocabulary
print("Chinese word-to-index dictionary:", word2idx_cn)  # Print the Chinese word-to-index dictionary
print("English word-to-index dictionary:", word2idx_en)  # Print the English word-to-index dictionary


# Define a function to randomly select a sentence and generate input, output, and target data from the vocabulary
def make_data(sentences):
    random_sentence = random.choice(sentences)
    # print(random_sentence)
    encoder_words = random_sentence[0].split()
    decoder_input_words = random_sentence[1].split()
    target_words = random_sentence[2].split()

    # # 确保 target 句子最后是 <eos>，并进行填充
    # target_words = target_words[:max_dec_len-1] + ['<eos>']  # 确保末尾有 <eos>
    # while len(target_words) < max_dec_len:
    #     target_words.append('<pad>')

    # while len(decoder_input_words) < max_dec_len:
    #     decoder_input_words.append('<pad>')

    encoder_input = np.array([[word2idx_cn[w] for w in encoder_words]])
    decoder_input = np.array([[word2idx_en[w] for w in decoder_input_words]])
    target = np.array([[word2idx_en[w] for w in target_words]])

    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)

    return encoder_input, decoder_input, target


# Use the make_data function to generate input, output, and target tensors
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences:  # Get the original sentence
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break
print("Original sentence:", original_sentence)  # Print the original sentence
print("Encoder input tensor shape:", encoder_input.shape)  # Print the input tensor shape
print("Decoder input tensor shape:", decoder_input.shape)  # Print the output tensor shape
print("Target tensor shape:", target.shape)  # Print the target tensor shape
print("Encoder input tensor:", encoder_input)  # Print the input tensor
print("Decoder input tensor:", decoder_input)  # Print the output tensor
print("Target tensor:", target)  # Print the target tensor

# Define the Attention class
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_context, encoder_context):
        scores = torch.matmul(decoder_context,encoder_context.transpose(-2,-1))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, encoder_context)
        return context, attn_weights

# Define the Encoder class, inheriting from nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # Set the size of the hidden layer
        self.embedding = nn.Embedding(input_size, hidden_size)  # Create an embedding layer
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # Create an RNN layer
    def forward(self, inputs, hidden):  # Forward propagation function
        embedded = self.embedding(inputs)  # Convert inputs to embedding vectors
        output, hidden = self.rnn(embedded, hidden)  # Input embedding vectors into the RNN layer and get the output
        return output, hidden
# Define the Decoder class, inheriting from nn.Module
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size  # Set the size of the hidden layer
        self.embedding = nn.Embedding(output_size, hidden_size)  # Create an embedding layer
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # Create an RNN layer
        self.out = nn.Linear(hidden_size, output_size)  # Create a linear output layer
    def forward(self, inputs, hidden):  # Forward propagation function
        embedded = self.embedding(inputs)  # Convert inputs to embedding vectors
        output, hidden = self.rnn(embedded, hidden)  # Input embedding vectors into the RNN layer and get the output
        output = self.out(output)  # Use the linear layer to generate the final output
        return output, hidden

# Define the docoder class
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention()
        self.out = nn.Linear(2 * hidden_size, output_size)
    def forward(self, dec_input, hidden, enc_output):
        embedded = self.embedding(dec_input)
        rnn_output, hidden = self.rnn(embedded, hidden)
        context, attn_weights = self.attention(rnn_output, enc_output)
        dec_output = torch.cat((rnn_output, context), dim=-1)
        dec_output = self.out(dec_output)
        return dec_output, hidden, attn_weights

n_hidden = 128  # Set the number of hidden layers
# Create the encoder and decoder
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print('Encoder structure:', encoder)  # Print the encoder structure
print('Decoder structure:', decoder)  # Print the decoder structure


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, hidden, trg=None, max_len=6, inference=False):
        encoder_outputs, hidden = self.encoder(src,hidden)
        batch_size = src.shape[0]
        outputs = []
        input_token = torch.tensor([[word2idx_en['<sos>']]] * batch_size)

        for t in range(max_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            if inference:
                input_token = output.argmax(2)
                if input_token.item() == word2idx_en['<eos>']:
                    break
            else:
                x = t + 1
                if x == len(trg[0]):
                    break
                input_token = trg[:, x].unsqueeze(1) if trg is not None else input_token
                # print(trg, t, input_token)
            
                
        outputs = torch.cat(outputs, dim=1)
        return outputs

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder    
    def forward(self, src, hidden, trg=None, max_len=6, inference=False): 
        encoder_outputs, hidden = self.encoder(src,hidden)
        batch_size = src.shape[0]
        outputs = []
        input_token = torch.tensor([[word2idx_en['<sos>']]] * batch_size)

        for t in range(max_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            if inference:
                input_token = output.argmax(2)
                if input_token.item() == word2idx_en['<eos>']:
                    break
            else:
                x = t + 1
                if x == len(trg[0]):
                    break
                input_token = trg[:, x].unsqueeze(1) if trg is not None else input_token
                # print(trg, t, input_token)
            
                
        outputs = torch.cat(outputs, dim=1)
        decoder_output, _, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_output) 
        return decoder_output, attn_weights



# Create the Seq2Seq architecture
model = Seq2Seq(encoder, decoder)
print('S2S model structure:', model)  # Print the model structure

# Define the training function
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences)  # Create training data
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)  # Initialize the hidden state
        optimizer.zero_grad()  # Clear gradients
        output = model(encoder_input, hidden, trg=decoder_input)  # Get the model output
        loss = criterion(output.view(-1, voc_size_en), target.view(-1))  # Calculate the loss
        if (epoch + 1) % 40 == 0:  # Print the loss
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
# Train the model
epochs = 400  # Number of training epochs
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer
train_seq2seq(model, criterion, optimizer, epochs)  # Call the function to train the model

plt.rcParams["font.family"]=['Songti SC']
plt.rcParams['font.sans-serif']=['Songti SC'] 
plt.rcParams['axes.unicode_minus']=False 
def visualize_attention(source_sentence, predicted_sentence, attn_weights):    
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(attn_weights, annot=True, cbar=False, 
                     xticklabels=source_sentence.split(), 
                     yticklabels=predicted_sentence, cmap="Greens") 
    plt.xlabel("源序列") 
    plt.ylabel("目标序列")
    plt.show() 

# Define the test function for the sequence-to-sequence model
def test_seq2seq(model, source_sentence):
    # Convert the input sentence to indices
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # Build the indices for the decoder input, starting with '<sos>' and followed by '<eos>' repeated for the length of the encoder input minus one
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']] * 5)

    # Convert to LongTensor type
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0)  # Add an extra dimension
    # print(encoder_input)
    # print(decoder_input)
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)  # Initialize the hidden state
    predict,attn_weights = model(encoder_input, hidden, decoder_input)  # Get the model output
    # print(predict)
    predict = predict.data.max(2, keepdim=True)[1]  # Get the index with the highest probability
    # Print the input sentence and the predicted sentence

    # print(predict)
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])

    attn_weights = attn_weights.squeeze(0).cpu().detach().numpy()
    print(attn_weights)
    visualize_attention(source_sentence, [idx2word_en[n.item()] for n in predict.squeeze()], attn_weights)


# Test the model
test_seq2seq(model, '我 爱 学习 人工智能')  # Test with the sentence "KaGe likes XiaoBing"
test_seq2seq(model, '自然 语言 处理 很 强大')  # Test with the sentence "NLP is so powerful"
test_seq2seq(model, '深度学习 改变 世界') 
test_seq2seq(model, '咖哥 喜欢 小冰')  
test_seq2seq(model, '神经网络 非常 复杂')   
