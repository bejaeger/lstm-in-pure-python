########################################
#
# Script to run entire chain 
# - Prepare Data
# - Build Model
# - Train Model and output generated text
#
#########################################
import src.dataprocessing as dp
import src.logger as log
from src.lstm import TwoLayerLSTM
from src.simplelstm import LSTM
import numpy as np


################################
# Settings

# For multi-layer LSTM
sequence_length = 100
n_neurons = 200

learning_rate = 0.003
n_epochs = 2

optimizer = "adam"
model_file_path = "models/adam-trained.npy"

# lrate for SGD
# learning_rate = 0.1


verbose=False

##############################################
# 1. Preprocessing
text = dp.get_data()
# log.info("First 50 words of input text: {}".format(text[0:50]))

# get unique set of charss
unique_chars = set(text)
# log.info("Found set of unique chars: {}".format(unique_chars))
# log.info("Length of unique chars: {}".format(len(unique_chars)))

# Get text as integer
char_to_index = {char:index for index,char in enumerate(unique_chars)}
index_to_char = {index:char for index,char in enumerate(unique_chars)}
text_as_int = np.array([char_to_index[c] for c in text])
# log.info("Transformed text to int: {}".format(text_as_int[0:50]))
# log.info("Length of text: {}".format(len(text_as_int)))

inputs, targets = dp.get_vectorized_and_shuffled_data(text_as_int, sequence_length)
# log.pp("First 10 entries in text: {}".format(text_as_int[0:10]))
# log.pp("First input: {}".format(inputs[0]))
# log.pp("First target: {}".format(targets[0]))
# log.pp("Second input: {}".format(inputs[1]))
# log.pp("Second target: {}".format(targets[1]))


###########################################
# 2. Model building and training

# model = LSTM(unique_chars=unique_chars, len_sequence=sequence_length, \
#     learning_rate=learning_rate, n_neurons=n_neurons, i_f_gate_coupled=False)

model = TwoLayerLSTM(unique_chars=unique_chars, len_sequence=sequence_length, \
    learning_rate=learning_rate, n_neurons=n_neurons, epochs=n_epochs,
    optimizer=optimizer, model_file_path=model_file_path, verbose=verbose
    )

model.train(inputs, targets)



