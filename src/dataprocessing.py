# Python module with functions for processing the data
# that is preprocessing (encoding) as well as 
# postprocessing (decoding)

# import requests
  
import requests
import os
import numpy as np

# We download the text that is already cleaned from: 
# https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
def get_data(path_to_file=None):
    data_path = "data/shakespeare.txt"
    if path_to_file:
        text = open(data_path, 'rb').read()
        text = str(text, 'utf-8')
    elif os.path.isfile(data_path):
        text = open(data_path, 'rb').read()
        text = str(text, 'utf-8')
    else:
        url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
        try:
            response = requests.get(url, allow_redirects=True)
            text = str(response.content, 'utf-8')
            print(type(text))
            with open(data_path, "w") as f:
                f.write(text)
        except Exception as e:
            print("ERROR: File with url '{}' could not be downloaded and written to disk".format(url))
            print(e)
    return text

def get_char_to_index_maps(unique_chars):
    char_to_index = {char:index for index,char in enumerate(unique_chars)}
    index_to_char = {index:char for index,char in enumerate(unique_chars)}
    return char_to_index, index_to_char

def get_text_as_int(text, unique_chars):
    char_to_index, index_to_char = get_char_to_index_maps(unique_chars)
    text_as_int = np.array([char_to_index[c] for c in text])
    return text_as_int

def get_vectorized_and_shuffled_data(int_text, sequence_length):
    # initialize data
    num_batches = len(int_text) // sequence_length
    int_text = int_text[: num_batches * sequence_length]  # trim to fit sequence length
    inputs = np.zeros((len(int_text) - sequence_length, sequence_length), dtype=np.int8)
    targets = np.zeros((len(int_text) - sequence_length, sequence_length), dtype=np.int8)
    for i in range(num_batches-1):
        step = i * sequence_length
        inputs[i] = int_text[step:sequence_length+step] # input
        targets[i] = int_text[step+1:sequence_length+step+1] # target
    p = np.random.permutation(num_batches - 1)
    return inputs[p], targets[p]
    
