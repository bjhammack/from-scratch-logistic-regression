import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

def sigmoid(z: float | ArrayLike) -> float | ArrayLike:
    '''
    Computes the sigmoid of z
    
    Args:
    z -- Scalar or numpy array

    Return:
    sigmoid -- the sigmoid of z
    '''
    sigmoid = 1/(1 + np.exp(-1 * z))
    return sigmoid


def propagate(
        X: ArrayLike,
        Y: ArrayLike,
        w: ArrayLike,
        b: float
        ) -> Tuple[ArrayLike, ArrayLike, float]:
    '''
    Calculates the cost function and its gradient.

    Args:
    X -- numpy array of size (n, m) where n = features and m = samples
    Y -- numpy array of size (1, m) containing the true binary labels
    w -- numpy array of size (n, 1) containing the weights
    b -- float representing the bias
    
    Return:
    cost -- negative log-likelihood cost for logistic regression function
    gradients -- dictionary with string keys and the below values
        dw -- numpy array shaped like w, gradient of the loss with respect to w
        db -- float representing the gradient of the loss with respect to b
    '''
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) + np.log(1-A))
    cost = np.squeeze(np.array(cost))
    
    dw = 1/m * np.dot(X,  (A - Y).T)
    db = 1/m * np.sum(A - Y)
    gradients = {'dw': dw, 'db': db}

    return cost, gradients

