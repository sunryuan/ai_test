# Construct a toy dataset
corpus = ["我特别特别喜欢看电影",
          "这部电影真的是很好看的电影",
          "今天天气真好是难得的好天气",
          "我今天去看了一部电影",
          "电影院的电影都很好看"]

# Tokenize sentences
import jieba  # Import jieba package
# Use jieba.cut for tokenization and convert the result to a list, stored in corpus_tokenized
corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]

# Create a vocabulary
word_dict = {}  # Initialize vocabulary
# Traverse the tokenized corpus
for sentence in corpus_tokenized:
    for word in sentence:
        # If the word is not in the vocabulary, add it to the vocabulary
        if word not in word_dict:
            word_dict[word] = len(word_dict)  # Assign current vocabulary index

print("Vocabulary:", word_dict)  # Print vocabulary

# Convert sentences to Bag-of-Words representation based on the vocabulary
bow_vectors = []  # Initialize Bag-of-Words representation
# Traverse the tokenized corpus
for sentence in corpus_tokenized:
    # Initialize a vector of all zeros, with a length equal to the vocabulary size
    sentence_vector = [0] * len(word_dict)
    for word in sentence:
        # Increment the count at the corresponding word index, indicating the word appears once in the current sentence
        sentence_vector[word_dict[word]] += 1
    # Add the current sentence's Bag-of-Words vector to the vector list
    bow_vectors.append(sentence_vector)

print("Bag-of-Words Representation:", bow_vectors)  # Print Bag-of-Words representation

# Import numpy library for calculating cosine similarity
import numpy as np

# Define cosine similarity function
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)  # Calculate the dot product of vectors vec1 and vec2
    norm_a = np.linalg.norm(vec1)  # Calculate the norm of vector vec1
    norm_b = np.linalg.norm(vec2)  # Calculate the norm of vector vec2
    return dot_product / (norm_a * norm_b)  # Return cosine similarity

# Initialize a matrix of all zeros to store cosine similarities
similarity_matrix = np.zeros((len(corpus), len(corpus)))

# Calculate cosine similarity between each pair of sentences
for i in range(len(corpus)):
    for j in range(len(corpus)):
        similarity_matrix[i][j] = cosine_similarity(bow_vectors[i],
                                                    bow_vectors[j])

# Import matplotlib library for visualizing cosine similarity matrix
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ['Songti SC']  # Set font style
plt.rcParams['font.sans-serif'] = ['Songti SC']  # Set sans-serif font style
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly

fig, ax = plt.subplots()  # Create a plot object

# Use matshow function to plot cosine similarity matrix with blue color map
cax = ax.matshow(similarity_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)  # Color bar for color mapping

ax.set_xticks(range(len(corpus)))  # X-axis ticks
ax.set_yticks(range(len(corpus)))  # Y-axis ticks
ax.set_xticklabels(corpus, rotation=45, ha='left')  # X-axis tick labels
ax.set_yticklabels(corpus)  # Y-axis tick labels

plt.show()  # Display the plot