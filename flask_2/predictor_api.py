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
import os

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

with app.open_resource('pickles/unique_chars.pkl', 'rb') as f:
    chars = pickle.load(f)

with app.open_resource('pickles/char_to_int.pkl', 'rb') as f:
    chars_to_int = pickle.load(f)


# define and load model:
seq_len = 35
n_vocab = 51

model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, n_vocab)))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
filepath_current = "pickles/weights-improvement-150-0.5860.hdf5"
APP_STATIC = os.path.join(APP_ROOT, filepath_current)
model.load_weights(APP_STATIC)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# define helper function to adjust by temperature:
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    np.seterr(divide = 'ignore') 
    return np.argmax(probas)


# define function to generate text
def generate_dish(seeds, lstm_model, temp, seq_len, n_vocab, output_len,
                 char_to_int_dict, chars):
  """
  Generate a dish based on a series of seeds and an LSTM model
  
  seeds: list of strings (each being seq_length long)
  lstm_model: model, the LSTM model to use
  temp: int, the temperature hyperparameter for LSTM
  seq_len: int, the length of the seed string
  n_vocab: int, number of unique characters
  output_len: int, length of string to output (number of characters to generate)
  char_to_int_dict: dict to convert characters to integers
  chars: list, sorted set of all possible characters
  """
  
  # Select a random pattern from seeds (the list of all patterns) 
  # as a random seed
  start_index = np.random.randint(0, len(seeds))
  orig_seed = seeds[start_index]
  
  # initialize dish as empty list
  dish = []
  seed = orig_seed
  print(seed)
  
  # generate characters one by one
  for i in range(output_len):
      sampled = np.zeros((1, seq_len, n_vocab))
      for t, char in enumerate(seed):
          sampled[0, t, chars_to_int[char]] = 1.

      preds = lstm_model.predict(sampled, verbose=0)[0]
      next_index = sample(preds, temp)
      next_char = chars[next_index]

      seed += next_char
      seed = seed[1:]

      dish.append(next_char)

  dish_gen = "".join(dish)
  return dish_gen


def main():
  result = generate_dish(seeds=seeds, lstm_model=model, temp=0.25, seq_len=seq_len, 
                         n_vocab=n_vocab, output_len=100, char_to_int_dict=chars_to_int, chars=chars)
  return result


# This section checks that the generation code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    print('hello world')
    print(main())

