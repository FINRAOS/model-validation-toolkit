import ray
import numpy as np
import public

from sklearn.utils import resample

from . import bias_variance_mse, get_values, train_and_predict


def _prepare_X_and_y(X_train_values, y_train_values, prepare_X, prepare_y_train):
    return prepare_X(X_train_values), prepare_y_train(y_train_values)


@public.add
def bias_variance_compute_parallel(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    prepare_X=lambda x: x,
    prepare_y_train=lambda x: x,
    iterations=200,
    random_state=None,
    decomp_fn=bias_variance_mse,
    fit_kwargs=None,
    predict_kwargs=None,
):
    r"""Compute the bias-variance decomposition in parallel

    Args:
        estimator (EstimatorWrapper): estimator wrapped with a class extending
            EstimatorWrapper
        X_train: features for training
        y_train: ground truth labels for training
        X_test: features for testing
        y_test: ground truth labels for testing
        prepare_X (function, optional): function to transform feature datasets before
            calling fit and predict methods
        prepare_y_train (function, optional): function to transform training ground
            truth labels before calling fit method
        iterations (int, optional): number of iterations for the training/testing
        random_state (int, optional): random state for bootstrap sampling
        decomp_fn (function, optional): bias-variance decomposition function
        fit_kwargs (dict, optional): kwargs to pass to the fit method
        predict_kwargs (dict, optional): kwargs to pass to the predict method

    Returns:
        (average loss, average bias, average variance, net variance)"""
    if predict_kwargs is None:
        predict_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    if isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state)

    X_train_values = get_values(X_train)
    y_train_values = get_values(y_train)
    X_test_values = get_values(X_test)
    X_test_prepared = prepare_X(X_test_values)

    if random_state is None:
        result = [
            bootstrap_train_and_predict_ray.remote(
                estimator,
                X_train_values,
                y_train_values,
                X_test_prepared,
                prepare_X,
                prepare_y_train,
                fit_kwargs,
                predict_kwargs,
            )
            for _ in range(iterations)
        ]
    else:
        result = [
            train_and_predict_ray.remote(
                estimator,
                *_prepare_X_and_y(
                    *resample(
                        X_train_values, y_train_values, random_state=random_state
                    ),
                    prepare_X,
                    prepare_y_train
                ),
                X_test_prepared,
                fit_kwargs,
                predict_kwargs
            )
            for _ in range(iterations)
        ]

    predictions = np.array(ray.get(result))

    y_test_values = get_values(y_test)

    return decomp_fn(predictions, y_test_values)


@ray.remote
def train_and_predict_ray(
    estimator,
    X_train_values,
    y_train_values,
    X_test_prepared,
    fit_kwargs=None,
    predict_kwargs=None,
):
    r"""Train an estimator and get predictions from it

    Args:
        estimator (EstimatorWrapper): estimator wrapped with a class extending
            EstimatorWrapper
        X_train_values: numpy array of features for training
        y_train_values: numpy array of ground truth labels for training
        X_test_prepared: features for testing which has been processed by prepare_X
            function
        fit_kwargs (dict, optional): kwargs to pass to the fit method
        predict_kwargs (dict, optional): kwargs to pass to the predict method

    Returns:
        predictions"""
    return train_and_predict(
        estimator,
        X_train_values,
        y_train_values,
        X_test_prepared,
        fit_kwargs=fit_kwargs,
        predict_kwargs=predict_kwargs,
    )


@ray.remote
def bootstrap_train_and_predict_ray(
    estimator,
    X_train_values,
    y_train_values,
    X_test_prepared,
    prepare_X=lambda x: x,
    prepare_y_train=lambda x: x,
    fit_kwargs=None,
    predict_kwargs=None,
):
    r"""Train an estimator using a bootstrap sample of the training data and get
    predictions from it

    Args:
        estimator (EstimatorWrapper): estimator wrapped with a class extending
            EstimatorWrapper
        X_train_values: numpy array of features for training
        y_train_values: numpy array of ground truth labels for training
        X_test_prepared: features for testing which has been processed by prepare_X
            function
        prepare_X (function, optional): function to transform feature datasets before
            calling fit and predict methods
        prepare_y_train (function, optional): function to transform train ground truth
            labels before calling fit method
        fit_kwargs (dict, optional): kwargs to pass to the fit method
        predict_kwargs (dict, optional): kwargs to pass to the predict method

    Returns:
        predictions"""
    if predict_kwargs is None:
        predict_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    X_sample, y_sample = resample(X_train_values, y_train_values)

    return train_and_predict(
        estimator,
        X_sample,
        y_sample,
        X_test_prepared,
        prepare_X,
        prepare_y_train,
        fit_kwargs,
        predict_kwargs,
    )
