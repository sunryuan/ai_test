import random


# Construct a toy dataset
corpus = ["我喜欢吃苹果",
          "我喜欢吃香蕉",
          "她喜欢吃葡萄",
          "他不喜欢吃香蕉",
          "他喜欢吃苹果",
          "她喜欢吃草莓"]

# Define a tokenization function to convert text into a list of individual characters
def tokenize(text):
    return [char for char in text]  # Split text into a list of characters

# Define a function to calculate N-Gram frequency
from collections import defaultdict, Counter  # Import required libraries
def count_ngrams(corpus, n):
    ngrams_count = defaultdict(Counter)  # Create a dictionary to store N-Gram counts
    for text in corpus:  # Traverse each text in the corpus
        tokens = tokenize(text)  # Tokenize the text
        for i in range(len(tokens) - n + 1):  # Traverse the tokenized result to generate N-Grams
            ngram = tuple(tokens[i:i+n])  # Create an N-Gram tuple
            prefix = ngram[:-1]  # Get the prefix of the N-Gram
            token = ngram[-1]  # Get the target character of the N-Gram
            ngrams_count[prefix][token] += 1  # Update the N-Gram count
    return ngrams_count

bigram_counts = count_ngrams(corpus, 2)  # Calculate Bigram frequency
print("Bigram Frequencies:")  # Print Bigram frequencies
for prefix, counts in bigram_counts.items():
    print("{}: {}".format("".join(prefix), dict(counts)))

# Define a function to calculate N-Gram probabilities
def ngram_probabilities(ngram_counts):
    ngram_probs = defaultdict(Counter)  # Create a dictionary to store N-Gram probabilities
    for prefix, tokens_count in ngram_counts.items():  # Traverse each N-Gram prefix
        total_count = sum(tokens_count.values())  # Calculate the total count for the current prefix
        for token, count in tokens_count.items():  # Traverse each N-Gram for the prefix
            ngram_probs[prefix][token] = count / total_count  # Calculate the probability of each N-Gram
    return ngram_probs

bigram_probs = ngram_probabilities(bigram_counts)  # Calculate Bigram probabilities
print("\nBigram Probabilities:")
for prefix, probs in bigram_probs.items():
    print("{}: {}".format("".join(prefix), dict(probs)))

# Define a function to generate the next token
def generate_next_token(prefix, ngram_probs):
    if not prefix in ngram_probs:  # If the prefix is not in N-Grams, return None
        return None
    next_token_probs = ngram_probs[prefix] 
    next_token = max(next_token_probs, key=next_token_probs.get)  # Choose the token with the highest probability as the next token
    return next_token

# Define a function to generate the next token
def generate_next_token_random(prefix, ngram_probs):
    if not prefix in ngram_probs: # If the prefix is not in N-Grams, return None
        return None
    next_token_probs = ngram_probs[prefix]   # Get the probabilities for the next token given the prefix
    next_token = random.choices(list(next_token_probs.keys()), weights=list(next_token_probs.values()))[0]
    return next_token

# Define a function to generate text based on a prefix
def generate_text(prefix, ngram_probs, n, length=10):
    tokens = list(prefix) 
    for _ in range(length - len(prefix)):
        next_token = generate_next_token_random(tuple(tokens[-(n-1):]), ngram_probs)
        if not next_token:
            break
        tokens.append(next_token)
    return "".join(tokens)

# Input a prefix and generate text
generated_text = generate_text("我", bigram_probs, 2)
print("\nGenerated Text:", generated_text)  # Print the generated text
generated_text = generate_text("他", bigram_probs, 2)
print("\nGenerated Text:", generated_text)  # Print the generated text