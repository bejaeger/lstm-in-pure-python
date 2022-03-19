import numpy as np


class Parameter:
    def __init__(self, r_size, c_size, weight_offset):

        # xavier initialization
        self.weight_sd = np.sqrt(1 / c_size)

        # weight matrix
        self.W = np.random.randn(
            r_size, c_size) * self.weight_sd + weight_offset
        # bias weight
        self.b = np.zeros((r_size, 1))

        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Parameters for adam/adagrad optimizer
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)

        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def reset_gradients(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def clip_gradients(self, clip_min, clip_max):
        np.clip(self.dW, clip_min, clip_max, out=self.dW)
        np.clip(self.db, clip_min, clip_max, out=self.db)

    def update_params_sgd(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def update_params_adagrad(self, learning_rate):
        self.mW += self.dW*self.dW
        self.W -= learning_rate * self.dW / (np.sqrt(self.mW) + 1e-8)
        self.mb += self.db*self.db
        self.b -= learning_rate * self.db / (np.sqrt(self.mb) + 1e-8)

    def update_params_adam(self, learning_rate, beta1, beta2, iteration):
        # dw, db are from current minibatch
        if iteration == 0: iteration = 1
        # moment 1
        self.mW = beta1 * self.mW + (1-beta1) * self.dW
        self.mb = beta1 * self.mb + (1-beta1) * self.db

        # moment 2
        self.vW = beta2 * self.vW + (1-beta2) * (self.dW**2)
        self.vb = beta2 * self.vb + (1-beta2) * (self.db**2)

        # bias correction
        mW_corr = self.mW / (1 - beta1**iteration)
        mb_corr = self.mb / (1 - beta1**iteration)
        vW_corr = self.vW / (1 - beta2**iteration)
        vb_corr = self.vb / (1 - beta2**iteration)
        
        # update weights and biases
        epsilon = 1e-8
        self.W -= learning_rate * (mW_corr / (np.sqrt(vW_corr) + epsilon))
        self.b -= learning_rate * (mb_corr / (np.sqrt(vb_corr) + epsilon))

    def save_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.W, allow_pickle=False)
            np.save(f, self.b, allow_pickle=False)

    def load_from_file(self, i, file_path):
        with open(file_path, 'rb') as f:
            for l in range(i):
                print("Skipping")
                _ = np.load(f)
                _ = np.load(f)
            print("HI")
            self.W = np.load(f)
            self.b = np.load(f)
