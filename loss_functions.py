import numpy as np

def cross_entropy(labels, scores):
    """Cross entropy function.
    -SUM{p(x).log(q(x)}
    """
    #return -np.sum(labels * np.log(scores))
    return -np.dot(labels, np.log(scores))
