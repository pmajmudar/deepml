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

def initial_weights(shape):
    # return a 3x1 matrix with random weights between -1 and 1, mean =-0
    return 2*np.random.random(shape) - 1

X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

#y = np.array([ [0, 0, 1, 1] ]).T
y = np.array([ [0, 1, 1, 0] ]).T

def train(X, y):
    # 3 input neurons
    # 4 neurons in the hidden layer
    # 1 output
    weights_0 = initial_weights((3,5))
    weights_1 = initial_weights((5,1))
    print weights_0
    layer_1 = None
    layer_0 = X
    for i in xrange(10000):

        #import ipdb; ipdb.set_trace()
        #forward prop

        # Outputs a value for every training example
        layer_1 = logit(np.dot(layer_0, weights_0))
        layer_2 = logit(np.dot(layer_1, weights_1))

        # error vector between predicted and training i.e. (t - y)
        layer_2_err = y - layer_2

        # error vector multiplied by derivative i.e. y.(1-y)(t-y)
        # element wise multiply
        layer_2_delta = layer_2_err * logit_deriv(layer_2)

        # Layer 1 contributions to layer 2 error (using weights)
        layer_1_err = layer_2_delta.dot(weights_1.T)
        layer_1_delta = layer_1_err * logit_deriv(layer_1)


        # Update weights - dot product acomplishes sum over training
        # examples
        weights_1 += np.dot(layer_1.T, layer_2_delta)
        weights_0 += np.dot(layer_0.T, layer_1_delta)

    print "Output"
    print layer_2

if __name__ == """__main__""":
    train(X, y)
