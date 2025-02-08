import torch
import random
import torch.nn as nn
import torch.optim as optim

# Construct a very simple dataset
sentences = ["I like toys", "I love dad", "I hate getting-hit", "I believe mom"]

# Join all sentences together, separate into words with spaces, and remove duplicates to build the vocabulary
word_list = list(set(" ".join(sentences).split()))
# Create a dictionary that maps each word to a unique index
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# Create a dictionary that maps each index to the corresponding word
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)  # Calculate the size of the vocabulary
print('Vocabulary:', idx_to_word) 
print('Vocabulary:', word_to_idx)  # Print the word to index mapping dictionary
print('Vocabulary size:', voc_size)  # Print the vocabulary size

# Model hyperparameters
n_step = 2  # Number of time steps (context length)
n_hidden = 4  # Hidden layer size
embedding_size = 2  # Embedding vector size
batch_size = 3  # Batch size

# Construct batch data
def make_batch():
    input_batch = []  # Define input batch list
    target_batch = []  # Define target batch list
    selected_sentences = random.sample(sentences, batch_size)  # Randomly select sentences
    max_length = 0  # Find the maximum length of input sequences
    
    for sen in selected_sentences:  # Traverse each sentence
        word = sen.split()  # Split the sentence into words
        input_data = [word_to_idx[n] for n in word[:-1]]  # Create input data
        target = word_to_idx[word[-1]]  # Create target data
        max_length = max(max_length, len(input_data))  # Update maximum length
        input_batch.append(input_data)  # Add input to input batch list
        target_batch.append(target)  # Add target to target batch list
    
    # Pad input sequences to the maximum length
    padded_input_batch = []
    for input_data in input_batch:
        padded_input = input_data + [0] * (max_length - len(input_data))  # Pad with zeros
        padded_input_batch.append(padded_input)
    
    input_batch = torch.LongTensor(padded_input_batch)  # Convert input data to tensor
    target_batch = torch.LongTensor(target_batch)  # Convert target data to tensor
    return input_batch, target_batch  # Return input batch and target batch data

input_batch, target_batch = make_batch()  # Generate batch data

# Convert each index value in the input batch data to the corresponding original word
input_words = []
for input_idx in input_batch:
    input_words.append([idx_to_word[idx.item()] for idx in input_idx if idx.item() != 0])
print("Input batch data corresponding original words:", input_words)
print("Target batch data:", target_batch)  # Print target batch data

# Convert each index value in the target batch data to the corresponding original word
target_words = [idx_to_word[idx.item()] for idx in target_batch]
print("Target batch data corresponding original words:", target_words)

# Define Neural Probability Language Model (NPLM)
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__()
        self.C = nn.Embedding(voc_size, embedding_size)  # Define a word embedding layer
        # First linear layer, input size is n_step * embedding_size, output size is n_hidden
        self.linear1 = nn.Linear(n_step * embedding_size, n_hidden)
        # Second linear layer, input size is n_hidden, output size is voc_size, i.e., vocabulary size
        self.linear2 = nn.Linear(n_hidden, voc_size)

    def forward(self, X):  # Define the forward propagation process
        # Input data tensor X has shape [batch_size, n_step]
        X = self.C(X)  # Pass X through the word embedding layer, shape becomes [batch_size, n_step, embedding_size]
        X = X.view(X.shape[0], -1)  # Reshape to [batch_size, n_step * embedding_size]
        # Pass through the first linear layer and apply the tanh function
        hidden = torch.relu(self.linear1(X))  # hidden tensor shape is [batch_size, n_hidden]
        # Pass through the second linear layer to get the output
        output = self.linear2(hidden)  # output shape is [batch_size, voc_size]
        return output  # Return the output result


class NPLM2(nn.Module):
    def __init__(self):
        super(NPLM2, self).__init__()
        self.C = nn.Embedding(voc_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, n_hidden, batch_first=True)  # use lstm
        self.linear = nn.Linear(n_hidden, voc_size) 
    def forward(self, X):  # 定义前向传播过程
        X = self.C(X)  # Pass X through the word embedding layer, shape becomes [batch_size, n_step, embedding_size]
        lstm_out, _ = self.lstm(X) # lstm_out has shape [batch_size, n_step, n_hidden]
        output = self.linear(lstm_out[:, -1, :]) # output has shape [batch_size, voc_size]
        return output

model = NPLM2()  # Create an instance of the Neural Probability Language Model
print('NPLM model structure:', model)  # Print the model structure

criterion = nn.CrossEntropyLoss()  # Define the loss function as cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Define the optimizer as Adam with a learning rate of 0.1

# Train the model
for epoch in range(5000):  # Set the number of training iterations
    optimizer.zero_grad()  # Clear the gradients of the optimizer
    input_batch, target_batch = make_batch()  # Create input and target batch data
    output = model(input_batch)  # Pass the input data through the model to get the output result
    loss = criterion(output, target_batch.view(-1))  # Calculate the loss value
    if (epoch + 1) % 1000 == 0:  # Print the loss value every 1000 iterations
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss.item()))
    loss.backward()  # Backpropagation to calculate gradients
    optimizer.step()  # Update model parameters

# Make predictions
input_strs = [["I", "hate"], ["I", "like"], ["I", "love"], ["I", "believe"]]  # Input sequences to predict
# Convert input sequences to corresponding indices
input_indices = [[word_to_idx[word] for word in seq] for seq in input_strs]
input_batch = torch.LongTensor(input_indices)  # Convert input sequence indices to tensor

# Predict for input sequences, take the category with the highest probability in the output
predict_all = model(input_batch).data
predict = predict_all.max(1)[1]
# Convert prediction result indices to corresponding words
print(predict_all)

predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]
print(predict_strs)
for input_seq, pred in zip(input_strs, predict_strs):
    print(input_seq, '->', pred)  # Print input sequences and prediction results


