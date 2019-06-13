"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

# standard imports:
import warnings
warnings.filterwarnings("ignore")

from joblib import dump, load
import numpy as np
import pandas as pd
import pickle

# LSTM imports:
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# open seeds, chars, and char_to_int dictionary:
with open('pickles/patterns.pkl', 'rb') as f:
    seeds = pickle.load(f)

with open('pickles/unique_chars.pkl', 'rb') as f:
    chars = pickle.load(f)

with open('pickles/char_to_int.pkl', 'rb') as f:
    chars_to_int = pickle.load(f)


# define and load model:
seq_len = 35
n_vocab = 51

model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, n_vocab)))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath_current = "pickles/weights-improvement-150-0.5860.hdf5"
model.load_weights(filepath_current)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()


# define helper function to adjust by temperature:
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    np.seterr(divide = 'ignore') 
    return np.argmax(probas)


# def make_prediction(text):
#     """
#     Input:
#     feature_dict: a dictionary of the form {"feature_name": "value"}

#     Function makes sure the features are fed to the model in the same order the
#     model expects them.

#     Output:
#     Returns (x_inputs, probs) where
#       x_inputs: a list of feature values in the order they appear in the model
#       probs: a list of dictionaries with keys 'name', 'prob'
#     """
#     text_vect = vectorizer.transform([text])
#     result = nmf.transform(text_vect)
#     return text, topics[np.argmax(result)]


# This section checks that the generation code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    print('hello world')
    # print('num_patterns: ', len(seeds))
    # print(chars)
    # print(chars_to_int)