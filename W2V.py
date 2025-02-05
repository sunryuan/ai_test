# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys


sentences = [
    "Kage is Teacher", 
    "Mazong is Boss", 
    "Niuzong is Boss",
    "Xiaobing is Student", 
    "Xiaoxue is Student"
]

words = ' '.join(sentences).split()
word_list = list(set(words))
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)

def create_skipgram_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:
                    data.append((word, neighbor))
    return data

def create_cbow_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for idx, word in enumerate(sentence):
            context = sentence[max(idx - window_size, 0): idx] + sentence[idx + 1: min(idx + window_size + 1, len(sentence))]
            if context:
                data.append((context, word))
    return data

skipgram_data = create_skipgram_dataset(sentences)
cbow_data = create_cbow_dataset(sentences)

def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor

class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)
    
    def forward(self, X):
        hidden = self.input_to_hidden(X)
        output = self.hidden_to_output(hidden)
        return output

class SkipGramWithEmbedding(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGramWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, voc_size)
    
    def forward(self, X):
        embedded = self.embedding(X)
        output = self.output_layer(embedded)
        return output

class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)
    
    def forward(self, X):
        hidden = self.input_to_hidden(X)
        output = self.hidden_to_output(hidden)
        return output

class CBOWWithEmbedding(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOWWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, voc_size)
    
    def forward(self, X):
        embedded = self.embedding(X).mean(dim=0, keepdim=True)
        output = self.output_layer(embedded)
        return output

def train_model(model_type="linear", method="skipgram", embedding_size=2, learning_rate=0.002, epochs=1000):
    print(f"Training {method.upper()} - {model_type} model..")
    start_time = time.time()

    dataset = skipgram_data if method == "skipgram" else cbow_data
    
    if method == "skipgram":
        model = SkipGram(voc_size, embedding_size) if model_type == "linear" else SkipGramWithEmbedding(voc_size, embedding_size)
    else:
        model = CBOW(voc_size, embedding_size) if model_type == "linear" else CBOWWithEmbedding(voc_size, embedding_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_values = []
    
    for epoch in range(epochs):
        loss_sum = 0
        for context, target in dataset:
            if method == "skipgram":
                X = one_hot_encoding(context, word_to_idx).float().unsqueeze(0) if model_type == "linear" else torch.tensor([word_to_idx[context]], dtype=torch.long)
                y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)
            else:
                X = sum(one_hot_encoding(w, word_to_idx) for w in context).float().unsqueeze(0) if model_type == "linear" else torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
                y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)
            
            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(dataset)}")
            loss_values.append(loss_sum / len(dataset))
    
    print(f"{method.upper()} - {model_type} Word Embeddings:")
    for word, idx in word_to_idx.items():
        vec = model.embedding.weight[idx].detach().numpy() if model_type == "embedding" else model.input_to_hidden.weight[:, idx].detach().numpy()
        print(f"{word}: {vec}")
    
    fig, ax = plt.subplots()
    for word, idx in word_to_idx.items():
        vec = model.embedding.weight[idx].detach().numpy() if model_type == "embedding" else model.input_to_hidden.weight[:, idx].detach().numpy()
        ax.scatter(vec[0], vec[1])
        ax.annotate(word, (vec[0], vec[1]), fontsize=12)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"It taks: {elapsed_time:.6f} s..")
    
    plt.title(f'2D {method.upper()} - {model_type} Word Embeddings')
    plt.xlabel('Vector Dimension 1')
    plt.ylabel('Vector Dimension 2')
    plt.show()
    
    return model

if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == "cbow" else "skipgram"
    model_type = sys.argv[2] if len(sys.argv) > 2  and sys.argv[2] == "embedding" else "linear"
    model = train_model(model_type=model_type, method=method)
