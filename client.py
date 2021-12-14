import warnings
import flwr as fl
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys

import utils

if __name__ == "__main__":

    filename = sys.argv[1]
    print(filename)
    # Load MNIST dataset from https://www.openml.org/d/554
    X_train, y_train = utils.load_data(filename)
    X_test, y_test = utils.load_data('test.csv')

    # Create Support Vector Machine Model
    model = svm.SVC(
        kernel='rbf', 
        C=10, 
        gamma=0.1,
        probability=True
    )

    # Define Flower client
    class SVMClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            """Sets the parameters of a sklean LogisticRegression model."""
            utils.set_model_params(model, parameters)

            model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return (model.dual_coef_, model.intercept_), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            """Sets the parameters of a sklean LogisticRegression model."""
            utils.set_model_params(model, parameters)

            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=SVMClient())