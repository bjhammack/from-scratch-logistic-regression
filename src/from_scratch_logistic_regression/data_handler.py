from glob import glob
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from typing import Tuple

def get_images(parent_dir: str, size: Tuple[int, int]
        ) -> Tuple[ArrayLike, ArrayLike]:
    '''
    Gets two sets of images, from two subfolders based on parent directory.
    
    Args:
    parent_dir -- String; holds path to dir that contains the two image subfolders
    size -- tuple of ints; indicates how big the pictures should be

    Return:
    images -- numpy array of shape (image, image count)
    labels -- numpy array of shape (image count, 1)
    '''
    label_names = []
    for folder in glob(parent_dir+'/*'):
        label_names.append(folder.split('\\')[-1])

    images = []
    labels = []
    for label in label_names:
        for file in glob(parent_dir+f'/{label}/*'):
            image = np.asarray(Image.open(file).resize(size))
            images.append(image)
            labels.append(1 if label == label_names[0] else 0)

    images = np.asarray(images)
    labels = np.asarray([labels])

    return images, labels


def stage_images(X: ArrayLike, Y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    '''
    Flattens, standardizes, and shuffles the images in preperation for modeling.
    '''
    X_flat = X.reshape(X.shape[0], -1).T
    X_standard = X_flat / 255.
    X_shuffle, Y_shuffle = _shuffle_X_Y(X_standard, Y)
    return X_shuffle, Y_shuffle


def _shuffle_X_Y(X: ArrayLike, Y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    '''
    '''
    assert len(X) == len(Y)
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]