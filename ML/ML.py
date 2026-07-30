import csv
import numpy as np

##
###  Matrix functions
##


# Scale a matrix by a scalar value (element wise multiplication)
def scale_matricies(mat, scalar):
    return np.multiply(mat, scalar)

# Sum two matricies together
def sum_matricies(mat, mat2):
    return np.add(mat, mat2)

# Subtract curr from orig
def calc_gradient(curr, orig):
    return np.subtract(curr, orig)


##
###  mlp class
##


class mlp:

    # initialize the model
    # read in the data in file fname
    def __init__(self):
        self.initialized = False
        self.momentum = 0.7


    def is_initialized(self):
        return self.initialized   


    # initialize the training model with random weights of dimensions _k and _h
    # returns self.initialized, false on error, true on success
    def init_training_random(self, fname, _k, _h):
        
        X = []
        labels = []
        X, labels = self.read_data(fname)

        # test if datafile was valid
        if np.size(labels) < 1:
            self.initialized = False
            return self.initialized

        # set class variables for X, labels, n, and d
        self.X = X
        self.labels = labels
        self.n, self.d = np.shape(X)

        # set k and h variables
        self.k = _k
        self.h = _h

        # randomly fill W and V
        np.random.seed(1)  # seed for reproducability
        self.V = (np.random.rand(self.h+1, self.k) * 0.02) - 0.01
        self.W = (np.random.rand(self.d+1, self.h) * 0.02) - 0.01

        # forward propogate to build Z and Y
        self.forward_propogate(self.X)

        self.initialized = True
        return self.initialized


    # initialize the training model with input weights matricies
    # returns self.initialized, false on error, true on success
    def init_training_model(self, fname, V, W):
        
        X = []
        labels = []
        X, labels = self.read_data(fname)

        # test if datafile is valid
        if np.size(labels) < 1:
            self.initialized = False
            return self.initialized
        
        # set class variables for X, labels, n, and d
        self.X = X
        self.labels = labels
        self.n, self.d = np.shape(X)

        # set the model's weights
        self.set_weights(V, W)

        # forward propogation to build Z and Y
        self.forward_propogate(self.X)

        self.initialized = True
        return self.initialized


    # train the MLP model
    # returns -1 on error, training error rate on success
    def train(self, eta, epochs):

        if not(self.initialized):
            return -1

        # forwards propogate, initialize Y and Z
        self.forward_propogate(self.X)
        err = error_func(self.Y, self.labels)

        # get weight updates for momentum
        old_dV, old_dW = self.backward_propogate(eta)

        for i in range(epochs):
            
            # backwards propogate
            dV, dW = self.backward_propogate(eta)

            # update with momentum
            dV = dV + self.momentum * old_dV
            dW = dW + self.momentum * old_dW

            # update old weights for momentum
            old_dV = dV
            old_dW = dW
            
            # update weights
            self.update_weights(dV, dW)

            # forwards propogate
            self.forward_propogate(self.X)

            # re-calc error, exit if the difference is too small
            err_upd = error_func(self.Y, self.labels)
            if(abs(err - err_upd) <= 0.2 and i >= 25):
                break
            err = err_upd
            
        # return the error rate in predictions
        return error_rate(self.Y, self.labels)
    

    # run the current model on validation data
    # returns -1 on error, validation error rate on success
    def validate(self, fname):

        if not(self.initialized):
            return -1

        # return error on no validation data
        X, labels = self.read_data(fname)
        if np.size(X) < 1 or np.size(X, 1) != self.d:
            return -1

        self.forward_propogate(X)
        return error_rate(self.Y, labels)

    
    # run the current model on labelless data to make predictions
    # returns -1 on error, array of predicitons on success
    def predict(self, fname):
        
        if not(self.initialized):
            return -1

        # return error on no prediciton data
        X, labels = self.read_data(fname)
        if np.size(X) < 1 or np.size(X, 1) != self.d:
            return -1

        # append the labels column (prediciton data won't have labels)
        X = np.append(X, labels)  

        self.forward_propogate(X)
        return np.argmax(self.Y, axis=1)


##
###  Weights functions
##


    # set the model's weights
    def set_weights(self, V, W):

        # set h and k
        h, k = np.shape(V)
        self.h = h - 1
        self.k = k

        # set W and V
        self.W = W
        self.V = V


    # get the model's current weights
    def get_weights(self):
        return self.V, self.W


    # update the model's weights
    def update_weights(self, dV, dW):
        self.V = self.V + dV
        self.W = self.W + dW

    # set momentum factor
    def set_momentum(self, _momentum):
        self.momentum = _momentum



    # forwards propogate, build Z and Y
    def forward_propogate(self, _X):
        _n, _d = np.shape(_X)

        _X = np.append(np.ones((_n, 1)), _X, axis=1)
        # hidden layer with ReLU activation (vectorized: equivalent to the
        # per-element ReLU but done in one numpy op)
        Z = np.maximum(np.dot(_X, self.W), 0)
        self.Z = np.append(np.ones((_n, 1)), Z, axis=1)

        # output layer with softmax (vectorized). Subtracting the per-row max
        # is a numerically-stable, mathematically-identical form of
        # exp(O_i) / sum_j exp(O_j).
        O = np.dot(self.Z, self.V)
        O = O - np.max(O, axis=1, keepdims=True)
        expO = np.exp(O)
        self.Y = expO / np.sum(expO, axis=1, keepdims=True)
    

    # backwards propogate, Return dV and dW
    def backward_propogate(self, eta):
        R = np.zeros((self.n, self.k))
        for k in range(self.k):
            R[:,k] = (self.labels == k)

        X = np.append(np.ones((self.n, 1)), self.X, axis=1)
        # Gradients are averaged over the n samples (the 1/n factor) so that a
        # stable learning rate does not depend on the shard size. Previously the
        # gradient was a raw sum over all samples, which made the effective step
        # ~n x larger and forced an impractically tiny eta.
        dV = eta * np.transpose((np.dot(np.transpose((R - self.Y)), self.Z))) / self.n

        # dW = eta*(((R - Y_pred)*V(2:end,:)' .* (X*W >= 0))'*X)';
        # (X*W >= 0) is the ReLU gradient mask, vectorized
        XW = (np.dot(X, self.W) >= 0)
        # Vt = np.transpose(self.V[1:,:])
        Vt = np.transpose(np.delete(self.V, 0, 0))
        dW = np.transpose(np.multiply(np.dot((R-self.Y),Vt), XW))
        dW = eta * np.transpose(np.dot(dW,X)) / self.n

        return dV, dW


    # read data
    def read_data(self, fname):
        X = []
        labels = []
        try:
            file = open(fname, 'r')
            data = csv.reader(file)
            for line in data:
                labels.append(int(line[-1]))
                X.append([int(item) for item in line[:-1]])
            X = np.array(X)
            labels = np.array(labels)
        except:
            print("Failed to open file %s" % fname)

        return X, labels



# ReLU activation function
def ReLU(x):
    if 0 > x:
        return 0
    return x


# Calculate current error of predictions
def error_func(Y, labels):
    _n, _k = np.shape(Y)
    R = np.zeros((_n, _k))
    for k in range(_k):
        R[:,k] = (labels == k)
    return -np.sum(np.multiply(R, np.log(Y+0.000001)))


# Calculate the % of wrongly classified samples
# forward propogate MUST be called first on X corresponding to labels
def error_rate(Y, labels):
    _n = np.shape(labels)
    return (np.sum(np.not_equal(np.argmax(Y, axis=1), labels)) / _n)[0]
