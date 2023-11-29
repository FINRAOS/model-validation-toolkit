class EstimatorWrapper:
    r"""This is a wrapper class that can be inherited to conform any estimator to the fit/predict interface"""

    def fit(self, X, y, **kwargs):
        r"""Train the estimator

        Args:
            X: features
            y: ground truth labels
            kwargs (optional): kwargs for use in training
        """
        pass

    def predict(self, X, **kwargs):
        r"""Get predictions from the estimator

        Args:
            X: features
            kwargs (optional): kwargs for use in predicting
        """
        pass
