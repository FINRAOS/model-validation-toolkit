from . import EstimatorWrapper


class PyTorchEstimatorWrapper(EstimatorWrapper):
    def __init__(
        self, estimator, optimizer_generator, loss_fn, fit_fn=None, predict_fn=None
    ):
        r"""Create a wrapper for a PyTorch estimator

        Args:
            estimator: PyTorch estimator instance
            optimizer_generator: generator function for the optimizer
            loss_fn: loss function
            fit_fn (optional): custom fit function to be called instead of default one
            predict_fn (optional): custom predict function to be called instead
                of default one

        Returns:
            self
        """
        self.estimator = estimator
        self.optimizer_generator = optimizer_generator
        self.optimizer = optimizer_generator(estimator)
        self.loss_fn = loss_fn
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn

    def fit(self, X, y, **kwargs):
        r"""Train the estimator

        Args:
            X: features
            y: ground truth labels
            kwargs (optional): kwargs for use in training

        Returns:
            self
        """
        self.estimator.apply(PyTorchEstimatorWrapper._reset_parameters)

        if self.fit_fn is not None:
            self.fit_fn(self, X, y, **kwargs)
            return self

        if kwargs.get("epochs") is None:
            epochs = 100
        else:
            epochs = kwargs.get("epochs")

        for i in range(epochs):
            loss = 0
            if kwargs.get("batch_size") is None:
                batch_size = len(y)
            else:
                batch_size = kwargs.get("batch_size")
            for j in range(0, len(y), batch_size):
                batch_start = j
                batch_end = j + batch_size
                X_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                prediction = self.estimator(X_batch)
                loss = self.loss_fn(prediction, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if kwargs.get("verbose"):
                print(f"epoch: {i:2} training loss: {loss.item():10.8f}")

        return self

    def predict(self, X, **kwargs):
        r"""Get predictions from the estimator

        Args:
            X: features
            kwargs (optional): kwargs for use in predicting

        Returns:
            self
        """
        if self.predict_fn is not None:
            return self.predict_fn(self, X, **kwargs)

        import torch

        prediction_list = []
        with torch.no_grad():
            for value in X:
                prediction = self.estimator(value)
                if len(prediction) > 1:
                    prediction_list.append(prediction.argmax().item())
                else:
                    prediction_list.append(prediction.item())
        return prediction_list

    def _reset_parameters(self):
        r"""Reset parameters of the estimator"""
        if hasattr(self, "reset_parameters"):
            self.reset_parameters()
