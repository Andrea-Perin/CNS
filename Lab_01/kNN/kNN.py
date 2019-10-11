# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class my_kNN:
    """
    This class implements a simple Nearest Neighbors algorithm
    """
    def __init__(self, n_neighbors):
        """
        n_neighbors: Number of neighbors to consider
        """
        # Just save the input parameter
        # TODO
        
    def fit(self, X, y):
        """
        X: feature matrix ([N x M], N is the number of samples, M is the number of features)
        y: label vector (N x 1)
        """
        # For a basic implementation of the kNN algorithm, you just need to save 
        # the training data, and the corresponding labels in the object
        # TODO
        
    def distance(self, x, X):
        """
        Euclidean distance of x from each row of X
        x: dimension 1 x 2
        X: dimension N x 2
        output: N x 1
        """
        # Evaluate the distance of x from each row of X
        # You can use a for cycle, but try to avoid it
        # NB: you can specify the axis for most of the numpy operation. 
        #     For example, "np.sum(X)" sums all the elements of X (it returns a scalar value),
        #     but "np.sum(X, axis=1)" performs the operation along the SECOND axis (index starts from 0)
        return # TODO
        
    def get_neighbors(self, x):
        """
        Given a single sample x=[x1, x2] (dimension: 1x2), return the class of the n nearest neighbors
        """
        # Evaluate the distance of x from the previously stored training data
        # Look for the N nearest neighbors (you may need np.argsort(), look at the doc)
        # Get the corresponding classes from the previously stored training labels
        return # TODO
        
    def evaluate(self, X):
        """
        Evaluate the class for each row of the matrix X
        X: dimension N x 2
        output: N x 1
        """
        # For each row of X, search the neighbors and select the most frequent one
        y = [] # Let's store each result in a list
        for x in X:
            # Get neighbors class
            nn_classes = self.get_neighbors(x)
            # Count the occurrences (check the doc of np.unique(), with return_counts=True )
            class_vec, counts = np.unique(nn_classes, return_counts=True)
            # Get the index of the most frequent class
            best_class_idx = counts.argmax()
            # Get the value of the most frequent class
            best_class = class_vec[best_class_idx]
            # Append the result to the list
            y.append(best_class)
        return np.array(y)
        
    
def plot_data(X, y, title):
    """
    X: data points
    y: labels
    """
    plt.figure(figsize=(8,6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Hex values of some colors
    for cl in range(4):
        mask = y == cl
        plt.scatter(X[mask, 0], X[mask, 1], color=colors[cl], label='Class %d' % cl)
    plt.axvline(0, color='black', linestyle='--')    
    plt.axhline(0, color='black', linestyle='--')    
    plt.title(title)    
    plt.xlabel('x1')    
    plt.ylabel('x2')    
    plt.legend(framealpha=1)
    
    
#%%        

if __name__ == '__main__':
    
    plt.close('all')
    
    #%% Generate some train data
    ## Simple bidimensional problem X = [x1, x2]
    ## Class 0 -> x1 > 0 , x2 > 0
    ## Class 1 -> x1 > 0 , x2 < 0
    ## Class 2 -> x1 < 0 , x2 < 0
    ## Class 3 -> x1 < 0 , x2 > 0
    # Number of training points
    N_train = 1000 
    # Generate random training points uniformly in [-5, 5]
    X_train = np.random.rand(N_train, 2)
    X_train = X_train * 10 - 5
    # Assign labels
    mask_class_0 = (X_train[:, 0] > 0) & (X_train[:, 1] > 0) 
    mask_class_1 = (X_train[:, 0] > 0) & (X_train[:, 1] < 0) 
    mask_class_2 = (X_train[:, 0] < 0) & (X_train[:, 1] < 0) 
    mask_class_3 = (X_train[:, 0] < 0) & (X_train[:, 1] > 0) 
    # Just a check (No overlap -> The sum of the masks must be a vector full of True, the product must be full of False)
    assert all(mask_class_0 + mask_class_1 + mask_class_2 + mask_class_3)
    assert not all(mask_class_0 * mask_class_1 * mask_class_2 * mask_class_3)
    y_train = np.zeros(N_train, dtype=np.int)
    y_train[mask_class_1] = 1
    y_train[mask_class_2] = 2
    y_train[mask_class_3] = 3
    
    ### Plot training data
    plot_data(X_train, y_train, title='Training points')

    #%% Generate some test data
    N_test = 100
    X_test = np.random.rand(N_test, 2)
    X_test = X_test * 10 - 5
    
    #%% Fit the model
    
    ### Inizialize the classifier (5 Nearest Neighbors)
    classifier = my_kNN(5)
    ### Fit
    classifier.fit(X_train, y_train)
    
    #%% Test the model
    y_test = classifier.evaluate(X_test)
    
    #%% Plot results
    plot_data(X_test, y_test, title='TEST')
    
    #%% Perform some tests
    result = classifier.evaluate([[1, 1], 
                                  [1, -1],
                                  [-1, -1],
                                  [-1, 1]])
    
    if all(result == [0, 1, 2, 3]):
        print('TEST PASSED!')
    else:
        print('TEST NOT PASSED!')