from . import EstimatorWrapper


class TensorFlowEstimatorWrapper(EstimatorWrapper):
    def __init__(self, estimator):
        r"""Create a wrapper for a TensorFlow estimator

        Args:
            estimator: TensorFlow estimator instance

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
        self._reset_weights()
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
        predictions = self.estimator.predict(X, **kwargs)
        prediction_list = []
        for prediction in predictions:
            if len(prediction) > 1:
                prediction_list.append(prediction.argmax().item())
            else:
                prediction_list.append(prediction.item())
        return prediction_list

    def _reset_weights(self):
        r"""Reset weights of the estimator"""
        import tensorflow as tf

        for layer in self.estimator.layers:
            if hasattr(layer, "kernel_initializer") and hasattr(layer, "kernel"):
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            if hasattr(layer, "bias_initializer") and hasattr(layer, "bias"):
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
            if hasattr(layer, "recurrent_initializer") and hasattr(
                layer, "recurrent_kernal"
            ):
                layer.recurrent_kernal.assign(
                    layer.recurrent_initializer(tf.shape(layer.recurrent_kernal))
                )
