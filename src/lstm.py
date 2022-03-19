import numpy as np
import src.logger as log
from src.math import sigmoid, dsigmoid, tanh, dtanh, softmax
from src.parameter import Parameter


class TwoLayerLSTM:
    def __init__(self, unique_chars, len_sequence=100, n_neurons=100, weight_offset=0.0,
                 learning_rate=0.01, epochs=5, beta1=0.9, beta2=0.999,
                 clip_grad_min=-1, clip_grad_max=1, optimizer="adam", verbose=False,
                    model_file_path="models/model.npy"):

        self.unique_chars = unique_chars # unique characters in the training data
        self.n_chars = len(unique_chars)
        self.n_neurons = n_neurons  # # of units in the hidden layer
        self.len_sequence = len_sequence  # #G of time steps
        self.verbose = verbose

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.char_to_index = {char: index for index,
                              char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index,
                              char in enumerate(unique_chars)}

        self.weight_offset = weight_offset
        # parameters for adam optimizer
        self.beta1 = beta1
        self.beta2 = beta2

        # clipping gradients
        self.clip_grad_min = clip_grad_min
        self.clip_grad_max = clip_grad_max

        # optimizer
        # can be any of 'adam', 'sgd', or 'adagrad'.
        # This is implemented in the Parameter class
        self.optimizer = optimizer

        # output file: weights are stored in here
        self.model_file_path = model_file_path

        # ---------------------------------------------
        # Weights and bias initialization

        # Note, we can combine the input weights and recurrent weights into one weight matrix
        # and concatenate the hidden state with the new input before multiplying
        # Sized of concatenated input and hidden_state
        self.n_concat = self.n_neurons + self.n_chars

        self.wi0 = Parameter(self.n_neurons, self.n_concat,
                             weight_offset=self.weight_offset)
        self.wo0 = Parameter(self.n_neurons, self.n_concat,
                             weight_offset=self.weight_offset)
        self.wc0 = Parameter(self.n_neurons, self.n_concat,
                             weight_offset=0)

        self.wi1 = Parameter(self.n_neurons, self.n_neurons *
                             2, weight_offset=self.weight_offset)
        self.wo1 = Parameter(self.n_neurons, self.n_neurons *
                             2, weight_offset=self.weight_offset)
        self.wc1 = Parameter(self.n_neurons, self.n_neurons *
                             2, weight_offset=0)

        self.wv = Parameter(self.n_chars, self.n_neurons,
                            weight_offset=self.weight_offset)

        # ----------------------------------------
        # init LSTM vectors
        (self.z0, self.h0, self.c0,
         self.i0, self.cbar0, self.o0) = {}, {}, {}, {}, {}, {}

        (self.z1, self.h1, self.c1,
         self.i1, self.cbar1, self.o1) = {}, {}, {}, {}, {}, {}

        # output
        self.v, self.yhat = {}, {}

        # in case we load a previously trained model these
        # will be the starting states
        self.loaded_h0 = {}
        self.loaded_c0 = {}
        self.loaded_h1 = {}
        self.loaded_c1 = {}

    def all_pars(self):
        return [self.wi0, self.wo0, self.wc0, self.wi1, self.wo1, self.wc1, self.wv]

    def forward_step(self, t, input, prev_h0, prev_c0, prev_h1, prev_c1):
        """
        t: timestep t
        input: dim-1 input vector (one-hot)
        prev_h0: hidden state from previous step
        prev_c0: cell state from previous step
        prev_h1, prev_c1: as above but for layer 1
        """

        # this is just for clean notation
        Wi0 = self.wi0.W
        Wo0 = self.wo0.W
        Wc0 = self.wc0.W
        bi0 = self.wi0.b
        bo0 = self.wo0.b
        bc0 = self.wc0.b
        Wi1 = self.wi1.W
        Wo1 = self.wo1.W
        Wc1 = self.wc1.W
        bi1 = self.wi1.b
        bo1 = self.wo1.b
        bc1 = self.wc1.b
        Wv = self.wv.W
        bv = self.wv.b

        # Layer 1
        self.z0[t] = np.vstack((prev_h0, input))
        self.i0[t] = sigmoid(np.dot(Wi0, self.z0[t]) + bi0)
        self.o0[t] = sigmoid(np.dot(Wo0, self.z0[t]) + bo0)
        self.cbar0[t] = tanh(np.dot(Wc0, self.z0[t]) + bc0)
        # CIFG variant with ft = 1 - it
        self.c0[t] = self.i0[t] * self.cbar0[t] + (1 - self.i0[t]) * prev_c0
        self.h0[t] = self.o0[t] * tanh(self.c0[t])

        # Layer 2
        self.z1[t] = np.vstack((prev_h1, self.h0[t]))
        self.i1[t] = sigmoid(np.dot(Wi1, self.z1[t]) + bi1)
        self.o1[t] = sigmoid(np.dot(Wo1, self.z1[t]) + bo1)
        self.cbar1[t] = tanh(np.dot(Wc1, self.z1[t]) + bc1)
        # CIFG variant with ft = 1 - it
        self.c1[t] = self.i1[t] * self.cbar1[t] + (1 - self.i1[t]) * prev_c1
        self.h1[t] = self.o1[t] * tanh(self.c1[t])

        # outputs/predictions
        self.v[t] = np.dot(Wv, self.h1[t]) + bv
        self.yhat[t] = softmax(self.v[t])  # softmax

        return self.yhat[t], self.h0[t], self.c0[t], self.h1[t], self.c1[t]

    def forward_pass(self, input, target):
        """
        inputs: character sequence as integer with length len_sequence (e.g. [1, 2, 1, 5,  1])
        targets: character sequence as integer with length len_sequence (e.g. [2, 1, 5, 1, 9])
        """

        # Steps:

        # 1. loop through each character
        # 2. Get one hot encoded input
        # 3. Take one forward step (calculating gates)

        assert input.shape == (self.len_sequence,)
        assert target.shape == (self.len_sequence,)

        loss = 0
        max_posterior = 0
        accuracy = 0

        prev_h0 = np.zeros((self.n_neurons, 1))
        prev_c0 = np.zeros((self.n_neurons, 1))
        prev_h1 = np.zeros((self.n_neurons, 1))
        prev_c1 = np.zeros((self.n_neurons, 1))

        # run over sequence of characters
        for t in range(self.len_sequence):
            one_hot_input = self.get_one_hot(input[t])

            yhat, prev_h0, prev_c0, prev_h1, prev_c1 = self.forward_step(t, one_hot_input,
                                                                         prev_h0, prev_c0, prev_h1, prev_c1)

            # get posterior at target value
            loss += - np.log(yhat[target[t], 0] + 1e-8)
            max_posterior += yhat[target[t], 0]
            add = 1 if target[t] == np.argmax(yhat) else 0
            accuracy += add

        loss /= self.len_sequence
        max_posterior /= self.len_sequence
        accuracy /= self.len_sequence
        # return metrics and last hidden state and cell state
        return loss, max_posterior, accuracy, prev_h0, prev_c0, prev_h1, prev_c1

    def backward_step(self, t, y, next_dh0, next_dc0, next_dh1,
                      next_dc1, prev_c0, prev_c1):
        """
        Backward step
        t: timestep t
        y: target (integer)
        next_dh0: derivative of hidden state from next (t+1) timestep
        next_dc0: derivative of cell state from next (t+1) timestep
        prev_c0: cell state of previous timestep (t-1)
        next_dh1, next_dc1, prev_c1: as above but for layer 1
        """

        # for cleaner notation
        z1 = self.z1[t]
        h1 = self.h1[t]
        c1 = self.c1[t]
        z1 = self.z1[t]
        i1 = self.i1[t]
        cbar1 = self.cbar1[t]
        o1 = self.o1[t]

        z0 = self.z0[t]
        h0 = self.h0[t]
        c0 = self.c0[t]
        z0 = self.z0[t]
        i0 = self.i0[t]
        cbar0 = self.cbar0[t]
        o0 = self.o0[t]

        dWi0 = self.wi0.dW
        dWo0 = self.wo0.dW
        dWc0 = self.wc0.dW
        dbi0 = self.wi0.db
        dbo0 = self.wo0.db
        dbc0 = self.wc0.db
        dWi1 = self.wi1.dW
        dWo1 = self.wo1.dW
        dWc1 = self.wc1.dW
        dbi1 = self.wi1.db
        dbo1 = self.wo1.db
        dbc1 = self.wc1.db
        dWv = self.wv.dW
        dbv = self.wv.db

        Wi0 = self.wi0.W
        Wo0 = self.wo0.W
        Wc0 = self.wc0.W
        Wi1 = self.wi1.W
        Wo1 = self.wo1.W
        Wc1 = self.wc1.W
        Wv = self.wv.W

        # --------------------------
        # Derivative of Loss
        yhat = self.yhat[t]

        dv = yhat.copy()
        dv[y] -= 1

        # -----------------------
        # Layer 1
        dWv += np.dot(dv, h1.T)
        dbv += dv

        dh1 = np.dot(Wv.T, dv)
        dh1 += next_dh1

        do1 = dh1 * tanh(c1)
        da_o1 = do1 * o1*(1-o1)
        dWo1 += np.dot(da_o1, z1.T)
        dbo1 += da_o1

        dc1 = dh1 * o1 * dtanh(c1)
        dc1 += next_dc1

        dc_bar1 = dc1 * i1
        da_c1 = dc_bar1 * (1-cbar1**2)
        dWc1 += np.dot(da_c1, z1.T)
        dbc1 += da_c1

        di1 = dc1 * cbar1 - dc1 * prev_c1
        da_i1 = di1 * i1*(1-i1)

        dWi1 += np.dot(da_i1, z1.T)
        dbi1 += da_i1

        dz1 = (np.dot(Wi1.T, da_i1)
               + np.dot(Wc1.T, da_c1)
               + np.dot(Wo1.T, da_o1))
        current_dh1 = dz1[:self.n_neurons, :]
        current_dc1 = (1-i1) * dc1

        # ---------------------------------
        # Layer 0

        # get derivative of input of layer 1 from second part of dz1
        dh0 = dz1[self.n_neurons:2*self.n_neurons, :]
        dh0 += next_dh0

        # TODO: Write function to perform the following
        # TODO: calculations as they are the same as above
        do0 = dh0 * tanh(c0)
        da_o0 = do0 * o0*(1-o0)
        dWo0 += np.dot(da_o0, z0.T)
        dbo0 += da_o0

        dc0 = dh0 * o0 * dtanh(c0)
        dc0 += next_dc0

        dc_bar0 = dc0 * i0
        da_c0 = dc_bar0 * (1-cbar0**2)
        dWc0 += np.dot(da_c0, z0.T)
        dbc0 += da_c0

        di0 = dc0 * cbar0 - dc0 * prev_c0
        da_i0 = di0 * i0*(1-i0)

        dWi0 += np.dot(da_i0, z0.T)
        dbi0 += da_i0

        dz0 = (np.dot(Wi0.T, da_i0)
               + np.dot(Wc0.T, da_c0)
               + np.dot(Wo0.T, da_o0))
        current_dh0 = dz0[:self.n_neurons, :]
        current_dc0 = (1-i0) * dc0

        return current_dh0, current_dc0, current_dh1, current_dc1

    def backward_propagation(self, target):
        """
        Backward propagation. 
        All gates have been calculated for each timestep in the sequence.
        We only need the target to compare it to the predicted value and
        start the backward propagation.
        """
        # initialize
        next_dh0 = np.zeros((self.n_neurons, 1))
        next_dc0 = np.zeros((self.n_neurons, 1))
        next_dh1 = np.zeros((self.n_neurons, 1))
        next_dc1 = np.zeros((self.n_neurons, 1))

        # backward propagation!
        for t in reversed(range(self.len_sequence)):
            prev_c0 = np.zeros((self.n_neurons, 1)) if t == 0 else self.c0[t-1]
            prev_c1 = np.zeros((self.n_neurons, 1)) if t == 0 else self.c1[t-1]
            next_dh0, next_dc0, next_dh1, next_dc1 = self.backward_step(
                t, target[t], next_dh0, next_dc0,
                next_dh1, next_dc1, prev_c0, prev_c1)

        self.clip_gradients()

    def train(self, inputs, targets):
        """
        Training the neural network
        inputs: vector of input character sequences already formatted as integers (e.g. [[9, 2, 8], [2, 4, 1], ...])
        targets: vector of target character sequences already formatted as integers (e.g. [[2, 8, 2], [4, 1, 8], ...])

        Steps:
        # Loop over epochs and sequences and then:
        # 1. Reset gates and gradients for each sequence
        # 2. Forward iteration (loop through sequence)
        # 3. Backward prop to get gradients (loop through sequence reversed)
        # 4. Update weights
        # 5. Print some performance metrics (loss, max_posterior, accuracy)
        # 6. Print generated text if performance above certain threshold
        """

        log.header("Start Training with {} Sequences for {} Epochs".format(
            len(inputs), self.epochs))

        iteration = 0
        cum_loss = []
        cum_posterior = []
        cum_accuracy = []
        for epoch in range(self.epochs):
            for input, target in zip(inputs, targets):

                self.reset_gates()
                self.reset_gradients()

                # Forward iteration
                loss, max_posterior, accuracy, prev_h0, \
                    prev_c0, prev_h1, prev_c1 = self.forward_pass(
                        input, target)
                cum_posterior.append(max_posterior)
                cum_loss.append(loss)
                cum_accuracy.append(accuracy)

                # backward propagation
                self.backward_propagation(target)

                iteration += 1

                # update of weights
                self.update_pars(iteration)

                if iteration % 100 == 0:
                    log.train("Iteration: {}".format(iteration))
                    log.info("Mean loss: {:.3f}".format(np.mean(cum_loss)))
                    log.info("Mean max posterior: {:.2f}".format(
                        np.mean(cum_posterior)))
                    log.info("Mean accuracy: {:.2f}".format(
                        np.mean(cum_accuracy)))
                    if self.verbose:
                        log.info("Learning rate: {:.3f}".format(
                            self.learning_rate))
                        log.info("Average weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                            self.wv.W.mean(), self.wc0.W.mean(), self.wo0.W.mean(), self.wi0.W.mean()))
                        log.info("Std weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                            self.wv.W.std(), self.wc0.W.std(), self.wo0.W.std(), self.wi0.W.std()))
                        log.info("Gradients std Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                            self.wv.dW.std(), self.wc0.dW.std(), self.wo0.dW.std(), self.wi0.dW.std()))
                        log.info("Gradients std Wc1, Wo1, Wf1, Wi1: {:.4f}, {:.4f}, {:.4f}".format(
                            self.wc1.dW.std(), self.wo1.dW.std(), self.wi1.dW.std()))
                    if np.mean(cum_posterior) > 0.35:
                        log.sample(self.sample(400, prev_h0, prev_c0, prev_h1, prev_c1))
                    cum_loss = []
                    cum_posterior = []
                    cum_accuracy = []
            log.header("End of Epoch {}!".format(epoch))
        self.save_weights(prev_h0, prev_c0, prev_h1, prev_c1)
        log.header("End Training")
    

    def update_pars(self, iteration):
        """
        Updating the weights based on calculated gradients
        """
        for w in self.all_pars():
            if self.optimizer == "adam":
                w.update_params_adam(self.learning_rate,
                                     self.beta1, self.beta2, iteration)
            elif self.optimizer == "sgd":
                w.update_params_sgd(self.learning_rate)
            elif self.optimizer == "adagrad":
                w.update_params_adagrad(self.learning_rate)

    def sample(self, sample_size, prev_h0, prev_c0, prev_h1, prev_c1):

        self.reset_gates()

        first_idx = np.random.randint(self.n_chars)
        input = self.get_one_hot(first_idx)

        sample_string = ""

        for t in range(sample_size):
            y_hat, prev_h0, prev_c0, prev_h1, prev_c1 = \
                self.forward_step(t, input, prev_h0, prev_c0, prev_h1, prev_c1)
            # get a random index within the probability distribution of y_hat(ravel())
            # if we take max value we will get repeated output.
            idx = np.random.choice(range(self.n_chars), p=y_hat.ravel())
            if not idx == np.argmax(y_hat.ravel()):
                idx = np.random.choice(range(self.n_chars), p=y_hat.ravel())
            input = self.get_one_hot(idx)
            # find the char with the sampled index and concat to the output string
            char = self.index_to_char[idx]
            sample_string += char
        return sample_string

    # ------------------------------------------------
    # helper methods
    def reset_gradients(self):
        for w in self.all_pars():
            w.reset_gradients()

    def reset_gates(self):
        self.z0,  self.h0, self.c0, \
            self.i0, self.cbar0, self.o0, = {}, {}, {}, {}, {}, {}
        self.z1, self.h1, self.c1, \
            self.i1, self.cbar1, self.o1, = {}, {}, {}, {}, {}, {}
        self.v, self.yhat = {}, {}

    def clip_gradients(self):
        # clip gradients to avoid exploding gradients
        # otherwise training will diverge at some point
        clip_min = -1
        clip_max = 1
        for w in self.all_pars():
            w.clip_gradients(clip_min, clip_max)

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_chars, 1), dtype=np.int8)
        one_hot[index] = 1  # Input character
        return one_hot

    def save_weights(self, h0, c0, h1, c1):
        with open(self.model_file_path, 'wb') as f:
            for w in self.all_pars():
                np.save(f, w.W)
                np.save(f, w.b)
            np.save(f, h0)
            np.save(f, c0)
            np.save(f, h1)
            np.save(f, c1)

    def load_weights(self):
        with open(self.model_file_path, 'rb') as f:
            for w in self.all_pars():
                w.W = np.load(f)
                w.b = np.load(f)
            self.loaded_h0 = np.load(f)
            self.loaded_c0 = np.load(f)
            self.loaded_h1 = np.load(f)
            self.loaded_c1 = np.load(f)
            
