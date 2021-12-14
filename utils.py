from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def load_data(filename) -> Dataset:

    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    return x, y

def train_with_dummie(model):
    X_dummie, y_dummie = load_data('dummie.csv')
    model.fit(X_dummie, y_dummie)
    return model

def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 3  # Thera are 3 classes of transport modes
    n_features = 24  # Number of features in dataset
    model = train_with_dummie(model)
    model.classes_ = np.array([i+1 for i in range(n_classes)])
    model.dual_coef_ = np.zeros((n_classes-1, (n_features-1)*n_classes))
    model.intercept_ = np.zeros((n_classes,))


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    
    try:
        return (model.dual_coef_, model.intercept_)
    except:
        set_initial_params(model)
        return (model.dual_coef_, model.intercept_)



def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    #model = train_with_dummie(model)
    model.dual_coef_ = params[0]
    model.intercept_ = params[1]
    return model








def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )