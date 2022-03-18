import numpy as np
import src.logger as log
import src.dataprocessing as dp
from src.math import sigmoid, dsigmoid, tanh, dtanh, softmax


class LSTM:
    def __init__(self, unique_chars, len_sequence, n_neurons=100, i_f_gate_coupled=True,
                 weight_init_sd=0.1, learning_rate=0.01):

        self.unique_chars = unique_chars
        # no. of unique characters in the training data
        self.n_chars = len(unique_chars)
        self.n_neurons = n_neurons  # no. of units in the hidden layer
        self.len_sequence = len_sequence  # no. of time steps, also size of mini batch

        self.learning_rate = learning_rate  # learning rate
        self.weight_sd = weight_init_sd  # Standard deviation of weights for initialization

        self.char_to_index = {char: index for index,
                              char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index,
                              char in enumerate(unique_chars)}

        # ---------------------------------------------
        # Weights and bias initialization

        # Input weights

        # Note, we can combine the input weights and recurrent weights into one weight matrix
        # and concatenate the hidden state with the new input before multiplying
        # Sized of concatenated input and hidden_state
        self.n_concat = self.n_neurons + self.n_chars

        self.i_f_gate_coupled = i_f_gate_coupled
        # forget gate
        if not self.i_f_gate_coupled:
            self.Wf = np.random.randn(
                self.n_neurons, self.n_concat) * self.weight_sd + 0.5
            self.bf = np.ones((self.n_neurons, 1))
        # input gate
        self.Wi = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd + 0.5
        self.bi = np.zeros((self.n_neurons, 1))
        # output gate
        self.Wo = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd + 0.5
        self.bo = np.zeros((self.n_neurons, 1))
        # cell gate
        self.Wc = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd
        self.bc = np.zeros((self.n_neurons, 1))

        # output
        self.Wv = np.random.randn(
            self.n_chars, self.n_neurons) * self.weight_sd + 0.5
        self.bv = np.zeros((self.n_chars, 1))

        # gradients corresponding to above weights
        (self.dWi, self.dbi, self.dWo, self.dbo,
         self.dWc, self.dbc, self.dWv, self.dbv) = np.zeros_like(self.Wi), np.zeros_like(self.bi), np.zeros_like(self.Wo), np.zeros_like(self.bo), np.zeros_like(self.Wc), np.zeros_like(self.bc), np.zeros_like(self.Wv), np.zeros_like(self.bv)

        # define LSTM vectors
        (self.z, self.yhat, self.h, self.c,
         self.i, self.cbar, self.o, self.v) = {}, {}, {}, {}, {}, {}, {}, {}

        if not self.i_f_gate_coupled:
            self.dWf, self.dbf = 0, 0
            self.f = {}

    def forward_step(self, t, input, prev_h, prev_c):
        """
        t: timestep t
        input: dim-1 input vector (one-hot)
        prev_h: hidden state from previous step
        prev_c: cell state from previous step
        """

        self.z[t] = np.vstack((prev_h, input))
        self.i[t] = sigmoid(np.dot(self.Wi, self.z[t]) + self.bi)
        self.o[t] = sigmoid(np.dot(self.Wo, self.z[t]) + self.bo)
        self.cbar[t] = tanh(np.dot(self.Wc, self.z[t]) + self.bc)
        if not self.i_f_gate_coupled:
            self.f[t] = sigmoid(np.dot(self.Wf, self.z[t]) + self.bf)
            self.c[t] = self.i[t] * self.cbar[t] + self.f[t] * prev_c
        else:
            # ft = 1 - it
            self.c[t] = self.i[t] * self.cbar[t] + (1 - self.i[t]) * prev_c
        self.h[t] = self.o[t] * tanh(self.c[t])
        self.v[t] = np.dot(self.Wv, self.h[t]) + self.bv
        self.yhat[t] = softmax(self.v[t])  # softmax

        return self.yhat[t], self.h[t], self.c[t]

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
        # del_loss = 0
        # run over sequence of characters
        for t in range(self.len_sequence):
            one_hot_input = self.get_one_hot(input[t])

            # input to next time step
            prev_h = np.zeros((self.n_neurons, 1)) if t == 0 else self.h[t-1]
            prev_c = np.zeros((self.n_neurons, 1)) if t == 0 else self.c[t-1]

            _ = self.forward_step(t, one_hot_input, prev_h, prev_c)

            # get posterior at target value
            loss += - np.log(self.yhat[t][target[t], 0])
            max_posterior += self.yhat[t][target[t], 0]
            add = 1 if target[t] == np.argmax(self.yhat[t]) else 0
            accuracy += add

        loss /= self.len_sequence
        max_posterior /= self.len_sequence
        accuracy /= self.len_sequence
        # last hidden state and cell state
        return loss, max_posterior, accuracy, self.h[self.len_sequence-1], self.c[self.len_sequence-1]

    def backward_step(self, t, y, next_dh, next_dc, prev_c):
        """
        Backward step
        t: timestep t
        y: target (integer)
        next_dh: derivative of hidden state from next (t+1) timestep
        next_dc: derivative of cell state from next (t+1) timestep
        prev_c: cell state of previous timestep (t-1)
        """

        # we leave out the t at the end of the labels for cleaner notation
        # remember: this is PER time step, i.e. per character

        z = self.z[t]
        yhat = self.yhat[t]
        h = self.h[t]
        c = self.c[t]
        z = self.z[t]
        if not self.i_f_gate_coupled:
            f = self.f[t]
        i = self.i[t]
        cbar = self.cbar[t]
        o = self.o[t]
        v = self.v[t]

        dv = yhat.copy()
        dv[y] -= 1

        self.dWv += np.dot(dv, h.T)
        self.dbv += dv

        dh = np.dot(self.Wv.T, dv)
        dh += next_dh

        do = dh * tanh(c)
        da_o = do * o*(1-o)
        self.dWo += np.dot(da_o, z.T)
        self.dbo += da_o

        dc = dh * o * dtanh(c)
        dc += next_dc

        dc_bar = dc * i
        da_c = dc_bar * (1-cbar**2)
        self.dWc += np.dot(da_c, z.T)
        self.dbc += da_c

        if not self.i_f_gate_coupled:
            di = dc * cbar
        else:
            di = dc * cbar - dc * prev_c
        da_i = di * i*(1-i)

        self.dWi += np.dot(da_i, z.T)
        self.dbi += da_i

        if not self.i_f_gate_coupled:
            df = dc * prev_c
            da_f = df * f*(1-f)
            self.dWf += np.dot(da_f, z.T)
            self.dbf += da_f

        dz = (np.dot(self.Wi.T, da_i)
              + np.dot(self.Wc.T, da_c)
              + np.dot(self.Wo.T, da_o))
        if not self.i_f_gate_coupled:
            dz += np.dot(self.Wf.T, da_f)

        current_dh = dz[:self.n_neurons, :]
        if not self.i_f_gate_coupled:
            current_dc = f * dc
        else:
            current_dc = (1-i) * dc
        return current_dh, current_dc

    def backward_propagation(self, target):
        """
        Backward propagation. 
        All gates have been calculated for each timestep in the sequence.
        We only need the target to compare it to the predicted value and
        start the backward propagation.
        """
        # initialize
        next_dh = np.zeros((self.n_neurons, 1))
        next_dc = np.zeros((self.n_neurons, 1))

        # backward propagation!
        for t in reversed(range(self.len_sequence)):
            prev_c = np.zeros((self.n_neurons, 1)) if t == 0 else self.c[t-1]
            next_dh, next_dc = self.backward_step(
                t, target[t], next_dh, next_dc, prev_c)

        self.clip_gradients()

    def train(self, inputs, targets):
        """
        Training the neural network
        inputs: vector of input character sequences already formatted as integers (e.g. [[9, 2, 8], [2, 4, 1], ...])
        targets: vector of target character sequences already formatted as integers (e.g. [[2, 8, 2], [4, 1, 8], ...])

        Steps:
        # Start loop over sequences and do the following 
        # 1. Reset gates and gradients for each sequence
        # 2. Forward iteration (loop through sequence)
        # 3. Backward prop to get gradients (loop through sequence reversed)
        # 4. Update weights
        # 5. Print some performance metrics (loss, max_posterior, accuracy)
        # 6. Print generated text if performance above certain threshold
        """

        log.header("Start Training")

        iteration = 0
        cum_loss = []
        cum_posterior = []
        cum_accuracy = []
        for input, target in zip(inputs, targets):

            self.reset_gates()
            self.reset_gradients()

            # Forward iteration
            loss, max_posterior, accuracy, prev_h, prev_c = self.forward_pass(
                input, target)
            cum_posterior.append(max_posterior)
            cum_loss.append(loss)
            cum_accuracy.append(accuracy)

            # backward propagation
            self.backward_propagation(target)

            self.update_weights()

            if iteration % 100 == 0:
                log.train("Iteration: {}".format(iteration))
                log.info("Mean loss: {:.3f}".format(np.mean(cum_loss)))
                log.info("Mean max posterior: {:.2f}".format(
                    np.mean(cum_posterior)))
                log.info("Mean accuracy: {:.2f}".format(np.mean(accuracy)))
                log.info("Learning rate: {:.3f}".format(self.learning_rate))
                log.info("Average weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(self.Wv.mean(), self.Wc.mean(), self.Wo.mean(),  self.Wi.mean()))
                log.info("Std weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(self.Wv.std(), self.Wc.std(), self.Wo.std(), self.Wi.std()))
                if np.mean(cum_posterior) > 0.3:
                    log.sample(self.sample(prev_h=prev_h,
                               prev_c=prev_c, sample_size=200))
                cum_loss = []
                cum_posterior = []

            iteration += 1

        log.header("End Training")

    def update_weights(self):
        """
        Updating the weights based on calculated gradients
        For now, this is just simple gradient descent
        """
        # simple gradient descent
        for w, dw in zip([self.Wi, self.bi,
                          self.Wo, self.bo, self.Wc, self.bc, self.Wv, self.bv],
                         [self.dWi, self.dbi,
                          self.dWo, self.dbo, self.dWc, self.dbc, self.dWv, self.dbv]
                         ):
            w -= self.learning_rate * dw
        if not self.i_f_gate_coupled:
            for w, dw in zip([self.Wf, self.bf], [self.dWf, self.dbf]):
                w -= self.learning_rate * dw
        # slowly decaying learning rate
        if self.learning_rate > 0.001:
            self.learning_rate *= 0.999

    def sample(self, prev_h, prev_c, sample_size):

        self.reset_gates()

        first_idx = np.random.randint(self.n_chars)
        input = self.get_one_hot(first_idx)

        sample_string = ""

        for t in range(sample_size):

            y_hat, prev_h, prev_c = self.forward_step(t, input, prev_h, prev_c)
            # get a random index within the probability distribution of y_hat(ravel())
            # if we take max value we will get repeated output.
            idx = np.random.choice(range(self.n_chars), p=y_hat.ravel())
            input = self.get_one_hot(idx)
            # find the char with the sampled index and concat to the output string
            char = self.index_to_char[idx]
            sample_string += char
        return sample_string

    # ------------------------------------------------
    # helper methods
    def reset_gradients(self):
        self.dWi, self.dbi, self.dWo, self.dbo, \
            self.dWc, self.dbc, self.dWv, self.dbv = 0, 0, 0, 0, 0, 0, 0, 0
        if not self.i_f_gate_coupled:
            self.dWf, self.dbf = 0, 0

    def reset_gates(self):
        self.z, self.yhat, self.h, self.c, \
            self.z, self.i, self.cbar, self.o, self.v = {}, {}, {}, {}, {}, {}, {}, {}, {}
        if not self.i_f_gate_coupled:
            self.f = {}

    def clip_gradients(self):
        # clip gradients to avoid exploding gradients
        # otherwise training will diverge at some point
        clip_min = -1
        clip_max = 1
        for w in [self.dWi, self.dbi, self.dWo, self.dbo,
                  self.dWc, self.dbc, self.dWv, self.dbv]:
            np.clip(w, clip_min, clip_max, out=w)
        if not self.i_f_gate_coupled:
            for w in [self.dWf, self.dbf]:
                np.clip(w, clip_min, clip_max, out=w)

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_chars, 1), dtype=np.int8)
        one_hot[index] = 1  # Input character
        return one_hot


class TwoLayerLSTM:
    def __init__(self, unique_chars, len_sequence, n_neurons=100,
                 weight_init_sd=0.2, learning_rate=0.01):

        self.unique_chars = unique_chars
        # no. of unique characters in the training data
        self.n_chars = len(unique_chars)
        self.n_neurons = n_neurons  # no. of units in the hidden layer
        self.len_sequence = len_sequence  # no. of time steps, also size of mini batch

        self.learning_rate = learning_rate  # learning rate

        self.weight_sd = weight_init_sd  # Standard deviation of weights for initialization

        self.char_to_index = {char: index for index,
                              char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index,
                              char in enumerate(unique_chars)}

        # ---------------------------------------------
        # Weights and bias initialization

        # Note, we can combine the input weights and recurrent weights into one weight matrix
        # and concatenate the hidden state with the new input before multiplying
        # Sized of concatenated input and hidden_state
        self.n_concat = self.n_neurons + self.n_chars

        # --------------------------------------
        # Gates layer 0
        # input gate
        self.Wi0 = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd + 0.5
        self.bi0 = np.zeros((self.n_neurons, 1))
        # output gate
        self.Wo0 = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd + 0.5
        self.bo0 = np.zeros((self.n_neurons, 1))
        # cell gate
        self.Wc0 = np.random.randn(
            self.n_neurons, self.n_concat) * self.weight_sd
        self.bc0 = np.zeros((self.n_neurons, 1))

        # ---------------------------------------
        # Gates layer 1
        # input gate
        self.Wi1 = np.random.randn(
            self.n_neurons, 2*self.n_neurons) * self.weight_sd + 0.5
        self.bi1 = np.zeros((self.n_neurons, 1))
        # output gate
        self.Wo1 = np.random.randn(
            self.n_neurons, 2*self.n_neurons) * self.weight_sd + 0.5
        self.bo1 = np.zeros((self.n_neurons, 1))
        # cell gate
        self.Wc1 = np.random.randn(
            self.n_neurons, 2*self.n_neurons) * self.weight_sd
        self.bc1 = np.zeros((self.n_neurons, 1))

        # ------------------------------------------
        # output
        self.Wv = np.random.randn(
            self.n_chars, self.n_neurons) * self.weight_sd + 0.5
        self.bv = np.zeros((self.n_chars, 1))

        # momentum for AdaGrad
        self.mWi0 = np.zeros_like(self.Wi0)
        self.mWc0 = np.zeros_like(self.Wc0)
        self.mWo0 = np.zeros_like(self.Wo0)
        self.mbi0 = np.zeros_like(self.bi0)
        self.mbc0 = np.zeros_like(self.bc0)
        self.mbo0 = np.zeros_like(self.bo0)

        self.mWi1 = np.zeros_like(self.Wi1)
        self.mWc1 = np.zeros_like(self.Wc1)
        self.mWo1 = np.zeros_like(self.Wo1)
        self.mbi1 = np.zeros_like(self.bi1)
        self.mbc1 = np.zeros_like(self.bc1)
        self.mbo1 = np.zeros_like(self.bo1)

        self.mWv = np.zeros_like(self.Wv)
        self.mbv = np.zeros_like(self.bv)

        # --------------------------------------
        # gradients
        (self.dWi0, self.dbi0, self.dWo0,
         self.dbo0, self.dWc0, self.dbc0) = 0, 0, 0, 0, 0, 0
        (self.dWi1, self.dbi1, self.dWo1,
         self.dbo1, self.dWc1, self.dbc1) = 0, 0, 0, 0, 0, 0
        self.dWv, self.dbv = 0, 0

        # ----------------------------------------
        # init LSTM vectors
        (self.z0, self.h0, self.c0,
         self.i0, self.cbar0, self.o0) = {}, {}, {}, {}, {}, {}

        (self.z1, self.h1, self.c1,
         self.i1, self.cbar1, self.o1) = {}, {}, {}, {}, {}, {}

        self.v, self.yhat = {}, {}

    def forward_step(self, t, input, prev_h0, prev_c0, prev_h1, prev_c1):
        """
        t: timestep t
        input: dim-1 input vector (one-hot)
        prev_h0: hidden state from previous step
        prev_c0: cell state from previous step
        prev_h1, prev_c1: as above but for layer 1
        """

        # Layer 1
        self.z0[t] = np.vstack((prev_h0, input))
        self.i0[t] = sigmoid(np.dot(self.Wi0, self.z0[t]) + self.bi0)
        self.o0[t] = sigmoid(np.dot(self.Wo0, self.z0[t]) + self.bo0)
        self.cbar0[t] = tanh(np.dot(self.Wc0, self.z0[t]) + self.bc0)
        self.c0[t] = self.i0[t] * self.cbar0[t] + (1 - self.i0[t]) * prev_c0
        self.h0[t] = self.o0[t] * tanh(self.c0[t])

        # Layer 2
        self.z1[t] = np.vstack((prev_h1, self.h0[t]))
        self.i1[t] = sigmoid(np.dot(self.Wi1, self.z1[t]) + self.bi1)
        self.o1[t] = sigmoid(np.dot(self.Wo1, self.z1[t]) + self.bo1)
        self.cbar1[t] = tanh(np.dot(self.Wc1, self.z1[t]) + self.bc1)
        self.c1[t] = self.i1[t] * self.cbar1[t] + (1 - self.i1[t]) * prev_c1
        self.h1[t] = self.o1[t] * tanh(self.c1[t])

        # outputs/predictions
        self.v[t] = np.dot(self.Wv, self.h1[t]) + self.bv
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
            loss += - np.log(yhat[target[t], 0])
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

        # we leave out the t at the end of the labels for cleaner notation
        # remember: this is PER time step, i.e. per character

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

        # --------------------------
        # Derivative of Loss
        yhat = self.yhat[t]
        v = self.v[t]

        dv = yhat.copy()
        dv[y] -= 1

        # -----------------------
        # Layer 1
        self.dWv += np.dot(dv, h1.T)
        self.dbv += dv

        dh1 = np.dot(self.Wv.T, dv)
        dh1 += next_dh1

        do1 = dh1 * tanh(c1)
        da_o1 = do1 * o1*(1-o1)
        self.dWo1 += np.dot(da_o1, z1.T)
        self.dbo1 += da_o1

        dc1 = dh1 * o1 * dtanh(c1)
        dc1 += next_dc1

        dc_bar1 = dc1 * i1
        da_c1 = dc_bar1 * (1-cbar1**2)
        self.dWc1 += np.dot(da_c1, z1.T)
        self.dbc1 += da_c1

        di1 = dc1 * cbar1 - dc1 * prev_c1
        da_i1 = di1 * i1*(1-i1)

        self.dWi1 += np.dot(da_i1, z1.T)
        self.dbi1 += da_i1

        dz1 = (np.dot(self.Wi1.T, da_i1)
               + np.dot(self.Wc1.T, da_c1)
               + np.dot(self.Wo1.T, da_o1))
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
        self.dWo0 += np.dot(da_o0, z0.T)
        self.dbo0 += da_o0

        dc0 = dh0 * o0 * dtanh(c0)
        dc0 += next_dc0

        dc_bar0 = dc0 * i0
        da_c0 = dc_bar0 * (1-cbar0**2)
        self.dWc0 += np.dot(da_c0, z0.T)
        self.dbc0 += da_c0

        di0 = dc0 * cbar0 - dc0 * prev_c0
        da_i0 = di0 * i0*(1-i0)

        self.dWi0 += np.dot(da_i0, z0.T)
        self.dbi0 += da_i0

        dz0 = (np.dot(self.Wi0.T, da_i0)
               + np.dot(self.Wc0.T, da_c0)
               + np.dot(self.Wo0.T, da_o0))
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
        # Start loop over sequences and do the following 
        # 1. Reset gates and gradients for each sequence
        # 2. Forward iteration (loop through sequence)
        # 3. Backward prop to get gradients (loop through sequence reversed)
        # 4. Update weights
        # 5. Print some performance metrics (loss, max_posterior, accuracy)
        # 6. Print generated text if performance above certain threshold
        """

        log.header("Start Training")

        iteration = 0
        cum_loss = []
        cum_posterior = []
        cum_accuracy = []
        for input, target in zip(inputs, targets):

            self.reset_gates()
            self.reset_gradients()

            # Forward iteration
            loss, max_posterior, accuracy, prev_h0, \
                prev_c0, prev_h1, prev_c1 = self.forward_pass(input, target)
            cum_posterior.append(max_posterior)
            cum_loss.append(loss)
            cum_accuracy.append(accuracy)

            # backward propagation
            self.backward_propagation(target)

            # update of weights
            self.update_weights()

            iteration += 1

            if iteration % 100 == 0:
                log.train("Iteration: {}".format(iteration))
                log.info("Mean loss: {:.3f}".format(np.mean(cum_loss)))
                log.info("Mean max posterior: {:.2f}".format(
                    np.mean(cum_posterior)))
                log.info("Mean accuracy: {:.2f}".format(np.mean(cum_accuracy)))
                log.info("Learning rate: {:.3f}".format(self.learning_rate))
                log.info("Average weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(self.Wv.mean(), self.Wc0.mean(), self.Wo0.mean(), self.Wi0.mean()))
                log.info("Std weights Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(self.Wv.std(), self.Wc0.std(), self.Wo0.std(), self.Wi0.std()))
                log.info("Gradients Wv, Wc, Wo, Wf, Wi: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(self.dWv.mean(), self.dWc0.mean(), self.dWo0.mean(), self.dWi0.mean()))                
                if np.mean(cum_posterior) > 0.4:
                    log.sample(self.sample(prev_h0, prev_c0,
                                           prev_h1, prev_c1, sample_size=200))
                cum_loss = []
                cum_posterior = []
                cum_accuracy = []

        log.header("End Training")

    def update_weights(self):
        """
        Updating the weights based on calculated gradients
        For now, this is just simple gradient descent
        """
        # TODO: Implement proper parameter handling

        self.mWi0 += self.dWi0*self.dWi0
        self.Wi0 -= self.learning_rate * self.dWi0 / np.sqrt(self.mWi0 + 1e-8)
        self.mWc0 += self.dWc0*self.dWc0
        self.Wc0 -= self.learning_rate * self.dWc0 / np.sqrt(self.mWc0 + 1e-8)
        self.mWo0 += self.dWo0*self.dWo0
        self.Wo0 -= self.learning_rate * self.dWo0 / np.sqrt(self.mWo0 + 1e-8)
        self.mbi0 += self.dbi0*self.dbi0
        self.bi0 -= self.learning_rate * self.dbi0 / np.sqrt(self.mbi0 + 1e-8)
        self.mbc0 += self.dbc0*self.dbc0
        self.bc0 -= self.learning_rate * self.dbc0 / np.sqrt(self.mbc0 + 1e-8)
        self.mbo0 += self.dbo0*self.dbo0
        self.bo0 -= self.learning_rate * self.dbo0 / np.sqrt(self.mbo0 + 1e-8)

        self.mWi1 += self.dWi1*self.dWi1
        self.Wi1 -= self.learning_rate * self.dWi1 / np.sqrt(self.mWi1 + 1e-8)
        self.mWc1 += self.dWc1*self.dWc1
        self.Wc1 -= self.learning_rate * self.dWc1 / np.sqrt(self.mWc1 + 1e-8)
        self.mWo1 += self.dWo1*self.dWo1
        self.Wo1 -= self.learning_rate * self.dWo1 / np.sqrt(self.mWo1 + 1e-8)
        self.mbi1 += self.dbi1*self.dbi1
        self.bi1 -= self.learning_rate * self.dbi1 / np.sqrt(self.mbi1 + 1e-8)
        self.mbc1 += self.dbc1*self.dbc1
        self.bc1 -= self.learning_rate * self.dbc1 / np.sqrt(self.mbc1 + 1e-8)
        self.mbo1 += self.dbo1*self.dbo1
        self.bo1 -= self.learning_rate * self.dbo1 / np.sqrt(self.mbo1 + 1e-8)

        self.mWv += self.dWv*self.dWv
        self.Wv -= self.learning_rate * self.dWv / np.sqrt(self.mWv + 1e-8)
        self.mbv += self.dbv*self.dbv
        self.bv -= self.learning_rate * self.dbv / np.sqrt(self.mbv + 1e-8)

        # # simple gradient descent
        # for w, dw in zip([self.Wi0, self.bi0, self.Wo0, self.bo0,
        #                      self.Wc0, self.bc0],
        #                     [self.dWi0, self.dbi0, self.dWo0, self.dbo0,
        #                      self.dWc0, self.dbc0],
        #                     ):
        #     w -= self.learning_rate * dw
        # for w, dw in zip([self.Wi1, self.bi1, self.Wo1, self.bo1,
        #                      self.Wc1, self.bc1, self.Wv, self.bv],
        #                     [self.dWi1, self.dbi1, self.dWo1, self.dbo1,
        #                      self.dWc1, self.dbc1, self.dWv, self.dbv],
        #                     ):
        #     w -= self.learning_rate * dw                            
        # slowly decaying learning rate
        # if self.learning_rate > 0.003:
        #     self.learning_rate *= 0.9995

    def sample(self, prev_h0, prev_c0, prev_h1, prev_c1, sample_size):

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
        self.dWi0, self.dbi0, self.dWo0, \
            self.dbo0, self.dWc0, self.dbc0, \
            self.dWi1, self.dbi1, self.dWo1, \
            self.dbo1, self.dWc1, self.dbc1, \
            self.dWv, self.dbv = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

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
        for w in [self.dWi0, self.dbi0, self.dWo0,
                  self.dbo0, self.dWc0, self.dbc0,
                  self.dWi1, self.dbi1, self.dWo1,
                  self.dbo1, self.dWc1, self.dbc1,
                  self.dWv, self.dbv]:
            np.clip(w, clip_min, clip_max, out=w)

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_chars, 1), dtype=np.int8)
        one_hot[index] = 1  # Input character
        return one_hot
