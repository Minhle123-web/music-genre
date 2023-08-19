import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import numpy as np

glove_path = './glove.6B.100d.txt' # Adjust the path accordingly
embedding_dim = 100


def load_glove(glove_path, embedding_dim):
    # Create a dictionary to store the word embeddings
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    # Create a word index and embedding matrix
    word_index = {}
    embedding_matrix = np.zeros((len(embeddings_index), embedding_dim))
    for i, (word, vector) in enumerate(embeddings_index.items()):
        word_index[word] = i
        embedding_matrix[i] = vector

    return word_index, torch.tensor(embedding_matrix)


word_index, embedding_matrix = load_glove(glove_path, embedding_dim)
word_index = word_index

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)  # Initialize with GloVe

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1, bidirectional=True)  # Use 8 LSTM layers and set bidirectional=True
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # Add a fully connected layer with 128 units
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(128, output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        output = output[:, -1, :]  # Get the output of the last time step of the last LSTM layer
        output = self.fc_layers(output)
        output = self.sigmoid(output)
        
        return output