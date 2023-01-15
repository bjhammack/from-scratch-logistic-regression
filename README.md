# From Scratch Logistic Regression


## Summary
This project was an exercise on staying fresh with the underlying processes and mathematics of modeling. In this case, a logistic regression model was built from the ground up, using only base python and numpy for the modeling steps.


## Structure
This project follows standard python project structure. At the highest level you will find all config and git files, the source code is stored in [src/from_scratch_logistic_regression/](/src/from_scratch_logistic_regression), the data used was stored in the project at `src/../data/images/*`. This folder obviously wasn't uploaded due to size, but you can download the data yourself from the source farther down; you will have to reorganize the data some and delete four grayscale images, but after that they will be ready for modeling.

The source code is organized into three python files.
1. [data_handler.py](/src/from_scratch_logistic_regression/data_handler.py) contains all the necessary functions to wrangle and prep the data.
2. [logistic_functions.py](/src/from_scratch_logistic_regression/logistic_functions.py) contains the functions that are used for LR processes (in particular, sigmoid activation and forward/backward propagation).
3. [model.py](/src/from_scratch_logistic_regression/model.py) contains the Model class that brings all functions together to perform the modeling from beginning to end.
There is also the [run.py](/src/from_scratch_logistic_regression/run.py) file, which is a quick and simple script that runs the model and produces a visualization of the cost timeline.


## Data
The data ([source](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification)) is a set of chihuahua and muffin images. The model's ultimate goal will be to try and determine if an image is a chihuahua or muffin.

- Chihuahua's - 3,196 images
- Muffin's - 2,717 images

![example-chihuahua](/images/example_chihuahua.jpg) ![example-muffin](/images/example_muffin.jpg)


## The Pipeline

### Model initialization / Data ingestion
When the Model class is first initialized (`model = Model(...)`), it accepts three optional arguments (X, Y, and data_dir).
- `X` and `Y` are a prepared set of data and labels for modeling. They need to be numpy arrays in the shapes of (features, samples) and (1, samples) respectively.
- `data_dir` is the parent directory of where the images to be modeled are stored.
    - NOTE: The parent directory needs to contain two folders, one for each class. These subfolders need to contain the two sets of images to be classified. The images need to be JPG or PNG files and be in the RBG format.
If `X` and `Y` are not given, the model will immediatley attempt to read data from the given `data_dir`. Otherwise, `X` and `Y` are used for modeling.

During data ingestion (if no data was given), the images are read in via PIL, resized to 100x100, labeled based on their folder of origin, and converted to numpy arrays. These new Xs and Ys are returned to the model, which will then send them to the cleaning function.

The cleaning function flattens the X array into the shape (n, m), where n = features and m = sample count, standardizes the data by dividing all pixels by 255, then shuffles the data (the labels are shuffled alongside, to ensure they still match).

After these final steps, `self.X` and `self.Y` are updated to these new datasets and modeling can begin.

### Modeling
After initialization, modeling can begin by calling `model.model(...)` (see the function for details on its arguments).

Modeling steps:
1. X and Y are split into the train and test sets
2. The weights and bias are initialized to a np.array of zeros and 0.0.
3. Parameter optimization begins.
    - For the given number of iterations, forward and backward propagation is performed on the data (vias the `propagate()` function), adjusting the weights and bias by the learning rate.
    - Ultimately, the final weights and bias, along with the gradients and the array of calculated costs, are returned.
4. Predictions are performed on the train and test sets.
    - The test set predictions are performed to calculate the "true" accuracy of the model. The train set predictions are performed as a sanity check that ensures the model actually learned to accurately classify the images (to some degree).
5. The final step is the model returning the metadata of the run inside a dictionary. This includes the array of costs, the array of predictions for both the Y train and test sets, the weights, the biad, the given learning rate, and the number of iterations.


## Results
For the sake of show-and-tell, a sample run was performed, below are the results.

### Parameters
- Split - 0.8
- Iterations - 2000
- Learning rate - 0.05

### Final Accuracy Results
1. Train: 65.4%
2. Test: 63.8%

![costs](/images/cost_results.png)


## Writeup
Given the simplicity of this logisitic regression model, the questionable quality of the selected dataset, and the use of the sigmoid function for activation (at the very least maybe tanh should be used instead) I'm somewhat impressed with the results. In particular, I'm satisfied with how it has avoided extreme overfitting, which would have been recognized if there was a very large disparity between the train and test accuracies.

Chihuahuas' made up 54% of the images, so a truly awful model would have just chosen all Chihuahuas and seen an accuracy of ~54%. This means, at the very least, we can say that this model is better than both a random guess and picking only the class with the largest dataset.

As a model, this project works, but stinks. As a proof of concept of a from scratch logistic regression model, I think it did pretty good.
