from copy import deepcopy
import data_handler as dh
from logistic_functions import propagate, sigmoid
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Tuple, List


class Model:
    def __init__(self,
            X: ArrayLike=None,
            Y: ArrayLike=None,
            split: float=0.8,
            data_location: str='/data/images/',
            ):
        '''
        Class initializes a model, along with its functions, and preps data for modeling.

        Args:
        X -- numpy array of size (n, m) or None
        Y -- numpy array of size (m, 1) or None
        split -- float; determines the train/test split
        data_location -- string; if specified, points to the location of the data
            Note: stored images must be subfolders inside specified folder, with
            the name of their subfolders the label they should have.
        '''
        self.X = X
        self.Y = Y
        self.split = split

        if data_location and not X:
            self.prep_data(data_location)

    def prep_data(self, loc: str):
        '''
        Prepares images for modeling by getting, cleaning, and staging.

        Args:
        loc -- String; directory of parent folder holding both image folders
        '''
        images, labels = dh.get_images(loc, size=(200,200))
        self.X, self.Y = dh.stage_images(images, labels)

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

        w = np.zeros((dim, 1))
        b = 0.0
        return w, b

    def optimize(
            self,
            X: ArrayLike,
            Y: ArrayLike,
            w: ArrayLike,
            b: float,
            iterations: int=100,
            learning_rate: float=0.01,
            verbose: bool=True,
            ) -> Tuple[Dict[str, ArrayLike], Dict[str, ArrayLike], List[float]]:
        '''
        Optimizes the weights and bias of the model by performing gradient descent.

        Args:
        X -- numpy array of shape (n, m), where n = feature count and m = samples
        Y -- numpy array of shape (1, m); contains the true labels for each sample
        w -- numpy array of shape (n, 1); contains the weights
        b -- float; represents the bias
        iterations -- int; number of loops to optimize w and b
        learning_rate -- float; the degree to which the gradient descent will update
        verbose -- bool; prints details of progress of optimization when True

        Return:
        parameters -- dictionary with str keys and the weights and bias as values
        gradients -- dictionary with str keys and the dweights and dbias as values
        costs -- list of costs calculated during optimization
        '''
        w = deepcopy(w)
        b = deepcopy(b)
        
        costs = []
        for i in range(iterations):
            gradients, cost = propagate(X, Y, w, b)
            dw = gradients['dw']
            db = gradients['db']
            w = w - (learning_rate * dw)
            b = b - (learning_rate * db)

            if i % 100 == 0:
                costs.append(cost)
                if verbose:
                    print(f'Iteration {i}: {cost}')

        parameters = {'w': w, 'b': b}
        gradients = {'dw': dw, 'db': db}

        return parameters, gradients, costs

    def predict(self, X: ArrayLike, w: ArrayLike, b: float) -> ArrayLike:
        '''
        Predicts the label of given samples.

        Args:
        X -- numpy array of size (n, m), where n = feature size and m = samples
        w -- numpy array of size (n, 1); contains the weights
        b -- float; represents the bias

        Return:
        Y_hat -- numpy array of size (1, m); contains the predictions made on X
        '''
        m = X.shape[1]
        Y_hat = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = sigmoid(np.dot(w.T, X) + b)

        Y_hat = np.where(A <= 0.5, 0, 1)

        return Y_hat

    def model(
            self,
            X_train: ArrayLike,
            Y_train: ArrayLike,
            X_test: ArrayLike,
            Y_test: ArrayLike,
            iterations: int=1000,
            learning_rate: float=0.1,
            verbose: bool=False,
            ) -> Dict[str, ArrayLike | List[float] | float | int]:
        '''
        Predicts the label of given samples.

        Args:
        X_train/X_test -- numpy array of size (n, m), where n = feature size 
        and m = samples
        Y_train/Y_test -- numpy array of size (1, m); contains the test and train labels
        iterations -- int; number of loops to optimize w and b
        learning_rate -- float; the degree to which the gradient descent will update
        verbose -- bool; prints details of progress of optimization when True

        Return:
        metadata -- dictionary containing information on the model
        '''
        w, b = self.initialize_params(X_train.shape[0])
        params, _, costs = self.optimize(X_train, Y_train, w, b, iterations,
                                         learning_rate, verbose)
        w = params['w']
        b = params['b']
        Y_hat_test = self.predict(X_test, w, b)
        Y_hat_train = self.predict(X_train, w, b)

        if verbose:
            print(f'Train Acc: {100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100}')
            print(f'Test Acc: {100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100}')

        metadata = {
            'costs': costs,
            'Y_hat_train': Y_hat_train,
            'Y_hat_test': Y_hat_test,
            'w': w,
            'b': b,
            'learning_rate': learning_rate,
            'iterations': iterations,
        }

        return metadata