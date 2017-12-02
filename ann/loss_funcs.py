import numpy as np

def log_loss(y, yhat):
    """Calculate Log Loss function, or cross-entropy.

    Defined as:
        L = - [ y*log(yhat) + (1-y)*log(1-yhat) ]
    """
    return -1 * ( y*np.log(yhat) + (1 - y)*np.log(1 - yhat) )
