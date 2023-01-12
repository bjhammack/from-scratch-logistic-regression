import numpy as np
from numpy.typing import ArrayLike
from typing import Dict


class Model:
    def __init__(self,
            X: ArrayLike,
            Y: ArrayLike,
            split: float=0.8,
            params: Dict[str, ArrayLike | float]={'w': None, 'b': 0.0},
            ):
        if not params['w']:
            self.w, self.b = self.initialize_params(X.shape[0])
        else:
            self.w = params['w']
            self.b = params['b']

    def initialize_params(self, dim):
        '''
        Initializes weights (w) as a numpy array of zeros (dim, 1) and bias (b)
        as 0.
    
        Args:
        dim -- size of the w array (dim, 1)

        Return:
        w -- weights in array of the shape (dim, 1)
        b -- bias initialized to 0
        '''

        self.w = np.zeros((dim, 1))
        self.b = 0.0
        return w, b
