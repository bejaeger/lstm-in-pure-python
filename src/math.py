import numpy as np

# Helper methods
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - tanh(x)**2

def softmax(v):
    return np.exp(v) / np.sum(np.exp(v)) #softmax