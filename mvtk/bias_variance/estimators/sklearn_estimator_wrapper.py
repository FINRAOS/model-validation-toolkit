from . import EstimatorWrapper


class SciKitLearnEstimatorWrapper(EstimatorWrapper):
    def __init__(self, estimator):
        r"""Create a wrapper for a Scikit-Learn estimator

        Args:
            estimator: Scikit-Learn estimator instance

        Returns:
            self
        """
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        r"""Train the estimator

        Args:
            X: features
            y: ground truth labels
            kwargs (optional): kwargs for use in training

        Returns:
            self
        """
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        r"""Get predictions from the estimator

        Args:
            X: features
            kwargs (optional): kwargs for use in predicting

        Returns:
            self
        """
        return self.estimator.predict(X, **kwargs)
