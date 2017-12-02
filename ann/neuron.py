
import numpy as np

np.random.seed(1)

def sigmoid(x):
    """Compute sigmoid function.

    Defined as 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmod(x):
    """The Derivative can be expressed in terms
    of the output of a sigmoid, s.

    Args:
        x

    Returns ds, the derivative of the sigmoid wrt to x.
    """
#    if y.any() < 0.0 or y.any() > 1.0:
#        raise Exception

    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def relu(x):
    """Rectified Linear Unit - ReLU"""
    pass

def d_relu(x):
    pass

def tanh(x):
    """Tanh function."""
    pass
