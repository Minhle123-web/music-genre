import pandas as pd
from collections import Counter
import numpy as np

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from flask import Flask, request, jsonify, render_template
import numpy as np
import model
from model import LSTMModel, word_index

app = Flask(__name__)


lstm_weights = './model_weights_lstm.pth'
embedding_dim = 100
hidden_dim = 128
genre_labels = ['Alternative', 'Alternative Metal', 'Alternative Rock', 'Ballad',
       'Country', 'Cover', 'Deutschland', 'East Coast', 'Electronic',
       'En Español', 'Folk', 'France', 'Hard Rock', 'Hip-Hop',
       'Indie Rock', 'Metal', 'Pop', 'Pop-Rock', 'R&B', 'Rap', 'Rock',
       'Soul', 'Soundtrack', 'Trap', 'UK', 'Vietnam']

max_sequence_length = 200

vocab_size = len(word_index)


tokenizer = get_tokenizer('basic_english')


model = LSTMModel(vocab_size, embedding_dim, hidden_dim, len(genre_labels))

#model.load_state_dict(torch.load(lstm_weights, map_location=torch.device('cpu')))


def preprocess_lyrics(lyrics):
    tokens = tokenizer(lyrics)
    
    # Convert the tokens to their indices in the vocabulary
    sequence_indices = [word_index[word] for word in tokens if word in word_index]
    
    # Pad or truncate the sequence
    if len(sequence_indices) > max_sequence_length:
        sequence_indices = sequence_indices[:max_sequence_length]
    else:
        sequence_indices += [0] * (max_sequence_length - len(sequence_indices))
    
    # Convert to a PyTorch tensor
    sequence_tensor = torch.tensor([sequence_indices])

    return sequence_tensor

def predict_genre_lstm(lyrics, model):
    
    sequence_tensor = preprocess_lyrics(lyrics)

    # If using a GPU, move the tensor to the GPU
    # if torch.cuda.is_available():
    #     sequence_tensor = sequence_tensor.cuda()
    #     model = model.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(lstm_weights, map_location=device))
    model = model.to(device)

    # Put the model in evaluation mode
    model.eval()

    # Get the model's prediction
    with torch.no_grad():
        output = model(sequence_tensor)
        
    genre_probs = {genre_labels[idx]: round(prob.item(), 2) for idx, prob in enumerate(output[0])}


    sorted_probs = {genre: prob for genre, prob in sorted(genre_probs.items(), key=lambda item: item[1], reverse=True)}

    top_5_genres = list(sorted_probs.items())[:5]

    result = [{"genre": genre, "probability": probability} for genre, probability in top_5_genres]

    return result


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    lyrics = request.json['lyric']
    result = predict_genre_lstm(lyrics, model)
    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


