# Build a corpus where each line contains three sentences: Chinese, English (decoder input), and the target English translation
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
# Deduplicate to get vocabularies without duplicate words
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))
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

import numpy as np  # Import numpy
import torch  # Import torch
import random  # Import random library
# Define a function to randomly select a sentence and generate input, output, and target data from the vocabulary
def make_data(sentences):
    # Randomly select a sentence for training
    random_sentence = random.choice(sentences)
    # Convert the input sentence to corresponding indices
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    # Convert the output sentence to corresponding indices
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    # Convert the target sentence to corresponding indices
    target = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])
    # Convert input, output, and target batches to LongTensor
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

import torch.nn as nn  # Import torch.nn library
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
n_hidden = 128  # Set the number of hidden layers
# Create the encoder and decoder
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print('Encoder structure:', encoder)  # Print the encoder structure
print('Decoder structure:', decoder)  # Print the decoder structure

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # Initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_input, hidden, dec_input):  # Define the forward propagation function
        # Pass the input sequence through the encoder and get the output and hidden state
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)
        # Pass the encoder's hidden state to the decoder as the initial hidden state
        decoder_hidden = encoder_hidden
        # Pass the decoder input (target sequence) through the decoder and get the output
        decoder_output, _ = self.decoder(dec_input, decoder_hidden)
        return decoder_output
# Create the Seq2Seq architecture
model = Seq2Seq(encoder, decoder)
print('S2S model structure:', model)  # Print the model structure

# Define the training function
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences)  # Create training data
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)  # Initialize the hidden state
        optimizer.zero_grad()  # Clear gradients
        output = model(encoder_input, hidden, decoder_input)  # Get the model output
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

# Define the test function for the sequence-to-sequence model
def test_seq2seq(model, source_sentence):
    # Convert the input sentence to indices
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # Build the indices for the decoder input, starting with '<sos>' and followed by '<eos>' repeated for the length of the encoder input minus one
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']] * (len(encoder_input[0]) - 1))
    # Convert to LongTensor type
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0)  # Add an extra dimension
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)  # Initialize the hidden state
    predict = model(encoder_input, hidden, decoder_input)  # Get the model output
    predict = predict.data.max(2, keepdim=True)[1]  # Get the index with the highest probability
    # Print the input sentence and the predicted sentence
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])

# Test the model
test_seq2seq(model, '咖哥 喜欢 小冰')  # Test with the sentence "KaGe likes XiaoBing"
test_seq2seq(model, '自然 语言 处理 很 强大')  # Test with the sentence "NLP is so powerful"
