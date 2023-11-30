import numpy as np
import pandas as pd
import public

from scipy import stats
from sklearn.utils import resample


@public.add
def get_values(x):
    r"""If argument is a Pandas dataframe, return 'values' numpy array from it.

    Args:
        x (Any): pandas dataframe or anything else

    Returns:
        if pandas dataframe - return 'values' numpy array
        otherwise - return itself

    """
    if isinstance(x, pd.DataFrame):
        return x.values
    else:
        return x


@public.add
def train_and_predict(estimator, X_train_values, y_train_values, X_test_prepared,
                      prepare_X=lambda x: x, prepare_y_train=lambda x: x,
                      fit_kwargs=None, predict_kwargs=None):
    r"""Train an estimator and get predictions from it

    Args:
        estimator (EstimatorWrapper): estimator wrapped with a class extending
            EstimatorWrapper
        X_train_values: numpy array of features for training
        y_train_values: numpy array of ground truth labels for training
        X_test_prepared: feature set for testing which has been processed by
            prepare_X function
        prepare_X (function, optional): function to transform feature datasets
            before calling fit and predict methods
        prepare_y_train (function, optional): function to transform train ground
            truth labels before calling fit method
        fit_kwargs (dict, optional): kwargs to pass to the fit method
        predict_kwargs (dict, optional): kwargs to pass to the predict method

    Returns:
        predictions"""
    if predict_kwargs is None:
        predict_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    X_sample_prepared = prepare_X(X_train_values)
    y_sample_prepared = prepare_y_train(y_train_values)

    estimator = estimator.fit(X_sample_prepared, y_sample_prepared, **fit_kwargs)
    predictions = estimator.predict(X_test_prepared, **predict_kwargs)

    return predictions


@public.add
def bootstrap_train_and_predict(estimator, X_train_values, y_train_values,
                                X_test_prepared, prepare_X=lambda x: x,
                                prepare_y_train=lambda x: x,
                                random_state=None, fit_kwargs=None,
                                predict_kwargs=None):
    r"""Train an estimator using a bootstrap sample of the training data and get
    predictions from it

    Args:
        estimator (EstimatorWrapper): estimator wrapped with a class extending
            EstimatorWrapper
        X_train_values: numpy array of features for training
        y_train_values: numpy array of ground truth labels for training
        X_test_prepared: feature set for testing which has been processed by prepare_X
            function
        prepare_X (function, optional): function to transform feature datasets before
            calling fit and predict methods
        prepare_y_train (function, optional): function to transform train ground
            truth labels before calling fit method
        random_state (int, optional): random state for bootstrap sampling
        fit_kwargs (dict, optional): kwargs to pass to the fit method
        predict_kwargs (dict, optional): kwargs to pass to the predict method

    Returns:
        predictions"""
    X_sample, y_sample = resample(X_train_values, y_train_values,
                                  random_state=random_state)

    return train_and_predict(estimator, X_sample, y_sample, X_test_prepared, prepare_X,
                             prepare_y_train, fit_kwargs, predict_kwargs)


@public.add
def bias_variance_mse(predictions, y_test):
    r"""Compute the bias-variance decomposition the mean squared error loss function

    Args:
        predictions: numpy array of predictions over the set of iterations
        y_test: numpy array of ground truth labels

    Returns:
        (average loss, average bias, average variance, net variance)"""
    pred_by_x = np.swapaxes(predictions, 0, 1)

    main_predictions = np.mean(predictions, axis=0)

    avg_bias = np.mean((main_predictions - y_test) ** 2)

    arr_loss = np.zeros(pred_by_x.shape[0], dtype=np.float64)
    arr_var = np.zeros(pred_by_x.shape[0], dtype=np.float64)
    for i in range(pred_by_x.shape[0]):
        arr_loss[i] = np.mean((pred_by_x[i] - y_test[i]) ** 2)
        arr_var[i] = np.mean((pred_by_x[i] - main_predictions[i]) ** 2)
    avg_loss = np.mean(arr_loss)
    avg_var = np.mean(arr_var)

    return avg_loss, avg_bias, avg_var, avg_var


@public.add
def bias_variance_0_1_loss(predictions, y_test):
    r"""Compute the bias-variance decomposition using the 0-1 loss function

    Args:
        predictions: numpy array of predictions over the set of iterations
        y_test: numpy array of ground truth labels

    Returns:
        (average loss, average bias, average variance, net variance)"""
    pred_by_x = np.swapaxes(predictions, 0, 1)

    main_predictions = stats.mode(predictions, axis=0, keepdims=True).mode[0]

    avg_bias = np.mean(main_predictions != y_test)

    arr_loss = np.zeros(pred_by_x.shape[0], dtype=np.float64)
    arr_var = np.zeros(pred_by_x.shape[0], dtype=np.float64)
    var_b = 0.0     # biased example contribution to avg_var
    var_u = 0.0     # unbiased example contribution to avg_var
    for i in range(pred_by_x.shape[0]):
        pred_true = np.sum(pred_by_x[i] == y_test[i])
        pred_not_main = np.sum(pred_by_x[i] != main_predictions[i])

        arr_loss[i] = (predictions.shape[0] - pred_true) / predictions.shape[0]
        arr_var[i] = pred_not_main / predictions.shape[0]

        if main_predictions[i] != y_test[i]:
            prb_true_given_not_main = pred_true / pred_not_main if pred_not_main != 0 \
                else 0
            var_b += (pred_not_main / predictions.shape[0]) * prb_true_given_not_main
        else:
            var_u += pred_not_main / predictions.shape[0]

    var_b /= pred_by_x.shape[0]
    var_u /= pred_by_x.shape[0]

    avg_loss = np.mean(arr_loss)
    avg_var = np.mean(arr_var)
    net_var = var_u - var_b

    return avg_loss, avg_bias, avg_var, net_var


@public.add
def bias_variance_compute(estimator, X_train, y_train, X_test, y_test,
                          prepare_X=lambda x: x,
                          prepare_y_train=lambda x: x,
                          iterations=200, random_state=None,
                          decomp_fn=bias_variance_mse,
                          fit_kwargs=None, predict_kwargs=None):
    r"""Compute the bias-variance decomposition in serial

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
    if fit_kwargs is None:
        fit_kwargs = {}
    if predict_kwargs is None:
        predict_kwargs = {}

    if isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state)

    predictions = np.zeros((iterations, y_test.shape[0]))

    X_train_values = get_values(X_train)
    y_train_values = get_values(y_train)
    X_test_values = get_values(X_test)
    X_test_prepared = prepare_X(X_test_values)

    for i in range(iterations):
        predictions[i] = bootstrap_train_and_predict(estimator, X_train_values,
                                                     y_train_values,
                                                     X_test_prepared,
                                                     prepare_X, prepare_y_train,
                                                     random_state,
                                                     fit_kwargs, predict_kwargs)

    y_test_values = get_values(y_test)

    return decomp_fn(predictions, y_test_values)
