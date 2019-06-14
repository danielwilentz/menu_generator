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
import random

# LSTM imports:
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# open seeds, chars, char_to_int dictionary, and unique words:
pattern_path = os.path.join(APP_ROOT, 'pickles', 'patterns.pkl')
with open(pattern_path, 'rb') as f:
    seeds = pickle.load(f)

unique_path = os.path.join(APP_ROOT, 'pickles', 'unique_chars.pkl')
with open(unique_path, 'rb') as f:
    chars = pickle.load(f)

char_int_path = os.path.join(APP_ROOT, 'pickles', 'char_to_int.pkl')
with open(char_int_path, 'rb') as f:
    chars_to_int = pickle.load(f)

unique_words_path = os.path.join(APP_ROOT, 'pickles', 'unique_words.pkl')
with open(unique_words_path, 'rb') as f:
    unique_words = pickle.load(f)

# import reprlib
# print('SEED:', reprlib.repr(seeds))
# print('CHARS:', reprlib.repr(chars))
# print('CHAR2INT:', reprlib.repr(chars_to_int))


# define and load model:
seq_len = 35
n_vocab = 51

model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, n_vocab)))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model_path = os.path.join(APP_ROOT, 'pickles', 'weights-improvement-150-0.5860.hdf5')
model.load_weights(model_path)
global graph
graph = tf.get_default_graph() 
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

def clean_output(raw_dish, avail_words):
    """
    function to clean the output of my LSTM
    """  
    
    bad_words = ['the', 'potat', 'nd', 'ad', 'comments', 'look', 'potatoe', 'read', 'er', 'us', 'review']
    filler_words = ['and', 'of', 'with', 'or', 'a', 'choice', 'of', 'served', 'on', 'cooked', 'in', 'from']
    
    word_list = raw_dish.strip().split(' ')[1:]
    cleaned_dish_list = []
    
    # Exclude any words not in the original corpus and in the "bad words list"
    # EXCEPT, I allow one typo to make things more interesting
    typos = 0
    for word in word_list:
        if word in avail_words and word not in bad_words:
            cleaned_dish_list.append(word)
        elif word not in avail_words:
            if typos < 1:
                cleaned_dish_list.append(word)
            typos += 1
    
    # Limit to only one and/one or/one sauce:
    num_and, num_or, num_sauce, num_potato, num_cheese = 0, 0, 0, 0, 0
    sauce_lim = random.randint(1,2)
    potato_lim = random.randint(1,2)
    cheese_lim = random.randint(1,2)
    
    for i, word in enumerate(cleaned_dish_list):
        if word == 'and':
            num_and += 1
        elif word == 'or':
            num_or += 1
        elif word == 'sauce':
            num_sauce += 1
        elif word == 'potato' or word == 'potatoes':
            num_potato += 1
        elif word == 'cheese':
            num_cheese += 1
        if num_and > 1 or num_or > 1 or num_sauce > sauce_lim or \
            num_potato > potato_lim or num_cheese > cheese_lim:
            cleaned_dish_list = cleaned_dish_list[:i]
    
    # Make sure the dish doesn't end with filler words
    while cleaned_dish_list[-1] in filler_words or len(cleaned_dish_list[-1]) <= 2:
        cleaned_dish_list = cleaned_dish_list[:-1]
    
    # Make sure the dish doesn't start with filler words
    while cleaned_dish_list[0] in filler_words:
        cleaned_dish_list = cleaned_dish_list[1:]
        
    return ' '.join(cleaned_dish_list)


# define function to generate text
def generate_dish(seeds, lstm_model, temp, seq_len, n_vocab, output_len,
                 char_to_int_dict, chars, avail_words):
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
#   print(seed)
  
  # generate characters one by one
  for w in range(output_len):
      sampled = np.zeros((1, seq_len, n_vocab))
      for t, char in enumerate(seed):
          sampled[0, t, chars_to_int[char]] = 1.

      with graph.as_default():
        preds = lstm_model.predict(sampled, verbose=0)[0]
      next_index = sample(preds, temp)
      next_char = chars[next_index]

      seed += next_char
      seed = seed[1:]

      dish.append(next_char)

  dish_gen = "".join(dish)
  clean_dish_gen = clean_output(dish_gen, avail_words)
  return clean_dish_gen


def main(temp):
  result = generate_dish(seeds=seeds, lstm_model=model, temp=temp, seq_len=seq_len, 
                         n_vocab=n_vocab, output_len=100, char_to_int_dict=chars_to_int, 
                         chars=chars, avail_words=unique_words)
  return result


# This section checks that the generation code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    print('hello world')
    print(main(temp=0.25))
