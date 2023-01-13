from copy import deepcopy
import data_handler as dh
from logistic_functions import propagate, sigmoid
from math import ceil
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Tuple, List


class Model:
    def __init__(self,
            X: ArrayLike=None,
            Y: ArrayLike=None,
            split: float=0.8,
            data_location: str='data/images/',
            ):
        '''
        Class initializes a model, along with its functions, and preps data
        for modeling.

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
        images, labels = dh.get_images(loc, size=(100,100))
        self.X, self.Y = dh.stage_images(images, labels)

    def train_test_split(self, X: ArrayLike, Y: ArrayLike, split: float=0.8
            ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        '''
        Splits pre-shuffled X and Y arrays based on the split value.

        Args:
        X -- numpy array of size (n, m)
        Y -- numpy array of size (1, m)
        split -- float; signifies the percentage size of the train set

        Return:
        X_train, Y_train, X_test, Y_test
        '''
        total_size = X.shape[0]
        train_size = ceil(total_size * split)

        X_train = X[:train_size]
        Y_train = Y[:train_size]
        X_test = X[train_size:]
        Y_test = Y[train_size:]

        assert len(X_train) + len(X_test) == total_size

        return X_train, Y_train, X_test, Y_test

    def initialize_params(self, dim: int) -> Tuple[ArrayLike, float]:
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
            cost, gradients = propagate(X, Y, w, b)
            dw = gradients['dw']
            db = gradients['db']
            w = w - (learning_rate * dw)
            b = b - (learning_rate * db)

            if i+1 % 100 == 0:
                costs.append(cost)
                if verbose: print(f'Iteration {i+1}: {cost}')

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
            X: ArrayLike,
            Y: ArrayLike,
            split: float=0.8,
            iterations: int=1000,
            learning_rate: float=0.5,
            verbose: bool=False,
            ) -> Dict[str, ArrayLike | List[float] | float | int]:
        '''
        Predicts the label of given samples.

        Args:
        X -- numpy array of size (n, m), where n = feature size and m = samples
        Y -- numpy array of size (1, m); contains the test and train labels
        split -- float; indicated how big you want the training sample to be
        iterations -- int; number of loops to optimize w and b
        learning_rate -- float; the degree to which the gradient descent will update
        verbose -- bool; prints details of progress of optimization when True

        Return:
        metadata -- dictionary containing information on the model
        '''
        if verbose: print(f'Splitting data into {split:.0%} training and {1-split:.0%} test sets.')
        X_train, Y_train, X_test, Y_test = self.train_test_split(X, Y, split)

        w, b = self.initialize_params(X_train.shape[0])
        if verbose: print(
            f'Optimizing weights and bias for {iterations} iterations and a '
            f'learning rate of {learning_rate}.')
        params, _, costs = self.optimize(X_train, Y_train, w, b, iterations,
                                         learning_rate, verbose)
        w = params['w']
        b = params['b']
        
        if verbose: print('Predicting test data.')
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
