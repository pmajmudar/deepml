import numpy as np

np.random.seed(1)

def logit(x):
    return 1 / (1 + np.exp(-x))

def logit_deriv(y):
    """The Derivative can be expressed in terms
    of the output of a logit, y.

    The inputs should thus be between 0 and 1.0
    """
#    if y.any() < 0.0 or y.any() > 1.0:
#        raise Exception

    return y*(1-y)

class NeuralNet(object):

    def __init__(self, hidden=2):
        # 3 input neurons
        # 4 neurons in the hidden layer
        # 1 output
        self.weights_0 = self._initial_weights((3,hidden))
        self.weights_1 = self._initial_weights((hidden,1))

    def _initial_weights(self, shape):
        # return a 3x1 matrix with random weights between -1 and 1, mean =-0
        return 2*np.random.random(shape) - 1

    def feedforward(self, X):
        X = np.array(X)
        rows,cols = X.shape
        layer_0 = np.ones((rows, cols+1))
        layer_0[:,:-1] = X

        # Outputs a value for every training example
        layer_1 = logit(np.dot(layer_0, self.weights_0))
        layer_2 = logit(np.dot(layer_1, self.weights_1))
        return layer_0, layer_1, layer_2

    def predict(self, X):
        _, _, y = self.feedforward(X)
        return y

    def train(self, X, y):
        for i in xrange(10000):
            #forward prop
            layer_0, layer_1, layer_2 = self.feedforward(X)

            # error vector between predicted and training i.e. (t - y)
            layer_2_err = y - layer_2

            # error vector multiplied by derivative i.e. y.(1-y)(t-y)
            # element wise multiply
            layer_2_delta = layer_2_err * logit_deriv(layer_2)

            # Layer 1 contributions to layer 2 error (using weights)
            layer_1_err = layer_2_delta.dot(self.weights_1.T)
            layer_1_delta = layer_1_err * logit_deriv(layer_1)

            # Update weights - dot product acomplishes sum over training
            # examples
            self.weights_1 += np.dot(layer_1.T, layer_2_delta)
            self.weights_0 += np.dot(layer_0.T, layer_1_delta)

        print "Output"
        print layer_2

X = np.array([ [0,0],
               [0,1],
               [1,0],
               [1,1] ])

#y = np.array([ [0, 0, 1, 1] ]).T

# XOR example
y = np.array([ [0, 1, 1, 0] ]).T

if __name__ == """__main__""":
    net = NeuralNet()
    net.train(X, y)
    predict = net.predict([[0,0.6]])
    print predict
