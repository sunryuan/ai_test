# -*- coding: utf-8 -*-

import torch  # Import torch library
import torch.nn as nn  # Import neural network module
import torch.optim as optim  # Import stochastic gradient descent optimizer
import matplotlib.pyplot as plt  # Import matplotlib for visualization

# Define a list of sentences to train CBOW and Skip-Gram models
sentences = [
    "Kage is Teacher", 
    "Mazong is Boss", 
    "Niuzong is Boss",
    "Xiaobing is Student", 
    "Xiaoxue is Student"
]

# Split sentences into words
words = ' '.join(sentences).split()

# Build a vocabulary list with unique words
word_list = list(set(words))

# Create a dictionary mapping each word to a unique index
word_to_idx = {word: idx for idx, word in enumerate(word_list)}

# Create a dictionary mapping each index to the corresponding word
idx_to_word = {idx: word for idx, word in enumerate(word_list)}

# Calculate vocabulary size
voc_size = len(word_list)

# Function to generate Skip-Gram training data
def create_skipgram_dataset(sentences, window_size=2):
    data = []  # Initialize data list
    for sentence in sentences:  # Iterate through each sentence
        sentence = sentence.split()  # Split sentence into a list of words
        for idx, word in enumerate(sentence):  # Iterate through words and their indices
            # Get neighboring words within the defined window size
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # Exclude the current word itself
                    # Append the word pair as training data
                    data.append((word, neighbor))
    return data

# Print results
print("Vocabulary:", word_list)
print("Word to Index Dictionary:", word_to_idx)
print("Index to Word Dictionary:", idx_to_word)
print("Vocabulary Size:", voc_size)

# Create Skip-Gram training data using the function
skipgram_data = create_skipgram_dataset(sentences)

# Print the first three samples of the unencoded Skip-Gram data
print("Skip-Gram Data Samples (Unencoded):", skipgram_data[:3])

# Define One-Hot encoding function
def one_hot_encoding(word, word_to_idx):    
    tensor = torch.zeros(len(word_to_idx))  # Create a zero tensor of the same length as the vocabulary
    tensor[word_to_idx[word]] = 1  # Set the value at the word's index position to 1
    return tensor  # Return the generated One-Hot encoded vector

# Example of One-Hot encoding
target_word = "Teacher"
print("Word before One-Hot Encoding:", target_word)
print("One-Hot Encoded Vector:", one_hot_encoding(target_word, word_to_idx))

# Display encoded Skip-Gram training data samples
print("Skip-Gram Data Samples (Encoded):", [(one_hot_encoding(target, word_to_idx), word_to_idx[context]) for context, target in skipgram_data[:3]])

# Define Skip-Gram model class
class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # Linear layer from vocabulary size to embedding size (weight matrix)
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)  
        # Linear layer from embedding size to vocabulary size (weight matrix)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)  
    
    def forward(self, X):  # Forward propagation, X shape is (batch_size, voc_size)      
        hidden = self.input_to_hidden(X)  # Hidden layer output, shape (batch_size, embedding_size)
        output = self.hidden_to_output(hidden)  # Output layer, shape (batch_size, voc_size)
        return output    

# Set embedding size, 2 is chosen for visualization purposes
embedding_size = 2 
skipgram_model = SkipGram(voc_size, embedding_size)  # Instantiate Skip-Gram model
print("Skip-Gram Model:", skipgram_model)

# Train Skip-Gram model
learning_rate = 0.001  # Set learning rate
epochs = 1000  # Set number of training epochs
criterion = nn.CrossEntropyLoss()  # Define cross-entropy loss function
optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)  # Initialize optimizer

# Start training loop
loss_values = []  # Store average loss per epoch
for epoch in range(epochs):
    loss_sum = 0  # Initialize loss sum
    for center_word, context in skipgram_data:        
        X = one_hot_encoding(center_word, word_to_idx).float().unsqueeze(0)  # Convert center word to One-Hot vector
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)  # Convert context word to index
        y_pred = skipgram_model(X)  # Compute predictions
        loss = criterion(y_pred, y_true)  # Compute loss
        loss_sum += loss.item()  # Accumulate loss
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
    if (epoch + 1) % 100 == 0:  # Output loss every 100 epochs
        print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(skipgram_data)}")  
        loss_values.append(loss_sum / len(skipgram_data))

# Plot training loss curve
plt.plot(range(1, epochs // 100 + 1), loss_values)  # Plot loss values
plt.title('Training Loss Curve')  # Set title
plt.xlabel('Epochs')  # Set x-axis label
plt.ylabel('Loss')  # Set y-axis label
plt.show()  # Display plot

