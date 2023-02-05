import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    proba = np.sum(y, axis = 0)/y.shape[0]

    EPS = 0.0005

    log_proba = np.log(proba + EPS)

    res =  -np.dot(proba, log_proba.T)
    # YOUR CODE HERE
    
    return res
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    proba = np.sum(y, axis = 0)/y.shape[0]
    res = 1 - np.sum(np.power(proba,2))
    # YOUR CODE HERE
    
    return res
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    y = np.power(y - np.mean(y), 2)
    # YOUR CODE HERE
    res = np.sum(y)/y.shape[0]
    return res

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    y = np.abs(y - np.median(y))
    # YOUR CODE HERE
    res = np.sum(y)/y.shape[0]
    return res


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
        
        
        
        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        left = X_subset[:, feature_index] < threshold
        right = X_subset[:, feature_index] >= threshold
        # YOUR CODE HERE

        X_left = X_subset[left, : ]
        y_left = y_subset[left, : ]

        X_right = X_subset[right, : ]
        y_right= y_subset[right, : ]

        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left = X_subset[:, feature_index] < threshold
        right = X_subset[:, feature_index] >= threshold

        y_left = y_subset[left, : ]

        y_right= y_subset[right, : ]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        H_m = self.criterion(y_subset)
        feature_index, threshold, best_result = -1, -1, -1e9
        for i in range(X_subset.shape[1]):
            values = np.sort(np.unique(X_subset[:, i]))
            for j in range(1, len(values)):
                left, right = self.make_split_only_y(i,values[j], X_subset, y_subset)
                H_l_n = self.criterion(left) * left.shape[0] / y_subset.shape[0]
                H_r_n = self.criterion(right) * right.shape[0] / y_subset.shape[0]
                res = H_m - H_l_n - H_r_n
                if res > best_result:
                    feature_index = i
                    threshold = values[j]
                    best_result = res

            

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth = 0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset

        def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        
        
        
        new_node = Node(-1, -1, proba=np.sum(y_subset,axis = 0) / y_subset.shape[0])
        if depth >= self.max_depth or X_subset.shape[0] < self.min_samples_split:
            return new_node
        if self.classification and np.max(np.sum(y_subset, axis = 0)) == y_subset.shape[0]:
            return new_node
        if not self.classification and np.unique(y_subset).shape[0] == 1: # debug
            return new_node
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        if feature_index == -1:
            return new_node
        new_node.feature_index = feature_index
        new_node.value = threshold
        left, right = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node.left_child = self.make_tree(left[0], left[1], depth + 1)
        new_node.right_child = self.make_tree(right[0], right[1], depth + 1)
        return new_node

    
           
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)


        self.root = self.make_tree(X, y)
    def predict_sample(self, x):
        """ Find and return predictions for all the samples in x """
        return np.array([self.predict_sample_help(x_i, self.root) for x_i in x])
    
    
    def predict_sample_help(self, x_i, node: Node):
        if x_i[node.feature_index] < node.value and not node.left_child is None:
            return self.predict_sample_help(x_i, node.left_child)
        elif x_i[node.feature_index] >= node.value and not node.right_child is None:
            return self.predict_sample_help(x_i, node.right_child)
        return node.proba

    def predict(self, X):

        # YOUR CODE HERE
        y_predicted = self.predict_sample(X)
        # print(y_predicted.shape,  y_predicted[1,:])
        if self.classification:
            y_predicted = np.argmax(y_predicted, axis=1)
        return y_predicted
        
    def predict_proba(self, X):
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = self.predict_sample(X)

        return y_predicted_probs

