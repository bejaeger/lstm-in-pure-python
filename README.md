# Train LSTM language model to generate text using pure Python

## Problem Setting
Q5 [Python (NumPy)] Implement a unidirectional multi-layer LSTM classifier with input and forget gates coupled. You can find information about this variant of LSTM [here](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer) (look for CIFG). The model should accept a feature vector as input and emit the corresponding posterior. Then train a character-based language model to generate text resembling Shakespeare (use any online dataset you see fit). How do you measure the quality of the generated text? Justify all the design decisions you’ve made in your training and inference pipelines.
NOTE: The implementation should be in pure NumPy and you are not allowed to use TF, PyTorch, Keras, etc.

## Run Training
Executing `python3 run/run.py` starts the training. If the 'mean max posterior' is above 0.35 the model generates text after every 100 iterations. 

## Overview of Solution
- Data preparation
  * Download shakespear dataset from [tensorflow.org](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)
  * Map chars to integers and vectorize data in input and target sequences
- Model building
  * An LSTM classs with two hidden layers (and input and forget gate coupled) is implemented in `src/lstm.py` (started out with simpler LSTM in `src/simplelstm.py`)
  * The other parameters are chosen based on manual optimization (sequence length, number neurons, learning rate)
  * I did not include peepholes as the paper states that they were not contributing to the performance significantly. 
- Model training
  * Cross-entropy is used as loss function
  * Implemented three optimizers: SGD, Adagrad, and Adam. Adam is chosen as default as it seems to converge the quickest
- Text Generation
  * Text is generated by choosing a random character seed and then randomly sampling from the posterior, given the previous hidden state and cell state
  * To improve the quality of the sampled text slightly, the random sampling is repeated once, if the chosen value does not correspond to the character with the highest posterior probability.
- How to measure the quality of text?
  * The training metrics are one indication of how well the model has understood the correlations in the data. However, they for example can't measure semantics or how well the text resembles the original adequately. 
  * One possibility is to have humans evaluate the text. This is, however, not very efficient.
  * Quantitative metrics are hard to construct and are a hot research topic. Most traditional methods try compare the generated text with the original (e.g. BLUE). More recently, evaluation metrics are used (and constructed) that model human judgement (e.g. [BLUERT](https://arxiv.org/abs/2004.04696)).
  * For the scope of this exercise I compared the generated text with the original to judge whether the training works.
  
## Example of generated text
```
LUCIO:
Her both by I , what love the sease that choose of the death,
And the sight the oath assurend mening so,

VALERIA:
Why, shall her danger ready to the despirest to him,
And thou faith, as you are now in your aged;
In York of it, Bortising traitor, my points
and Rome?

ISABELLA:
God you are for the provides the body.
```

## Resources
- I used the following papers/articles as resources: [LSTM: A Search Space Odyssey]([https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer]), [Deriving the backpropagation equations for a LSTM](https://christinakouridi.blog/2019/06/19/backpropagation-lstm/#:~:text=Backpropagation%20through%20a%20LSTM%20is,recursively%20applying%20the%20chain%20rule.), [Vanilla LSTM with numpy](https://blog.varunajayasiri.com/numpy_lstm.html)
- In the `notes/` directory I added scans of my own derivations of the forward pass and backpropagation for multi-layer LSTMs. 