import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge

from mvtk.bias_variance import (
    bias_variance_compute_parallel,
    bias_variance_mse,
    bias_variance_0_1_loss,
)
from mvtk.bias_variance.estimators import SciKitLearnEstimatorWrapper


def create_data():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.concatenate((np.arange(3), np.arange(3)), axis=None)
    X_test = np.arange(6).reshape(3, 2)
    y_test = np.array([0, 1, 1])

    return X_train, y_train, X_test, y_test


def test_bias_variance_compute_parallel_mse():
    X_train, y_train, X_test, y_test = create_data()

    model = Ridge(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_compute_parallel(
        model_wrapped,
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=123,
        decomp_fn=bias_variance_mse,
    )

    assert (np.round(avg_loss, decimals=12) ==
            np.round(np.float64(0.3967829075484304), decimals=12))
    assert (np.round(avg_bias, decimals=12) ==
            np.round(np.float64(0.13298143583764407), decimals=12))
    assert (np.round(avg_var, decimals=12) ==
            np.round(np.float64(0.26380147171078644), decimals=12))
    assert (np.round(net_var, decimals=12) ==
            np.round(np.float64(0.26380147171078644), decimals=12))

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)
    assert avg_var == net_var


def test_bias_variance_calc_parallel_0_1():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_compute_parallel(
        model_wrapped,
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=123,
        decomp_fn=bias_variance_0_1_loss,
    )

    assert avg_loss == np.float64(0.4566666666666666)
    assert avg_bias == np.float64(0.3333333333333333)
    assert avg_var == np.float64(0.33499999999999996)
    assert net_var == np.float64(0.12333333333333332)

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)


def test_bias_variance_calc_parallel_mse_no_random_state():
    X_train, y_train, X_test, y_test = create_data()

    model = Ridge(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_compute_parallel(
        model_wrapped,
        X_train,
        y_train,
        X_test,
        y_test,
        iterations=10,
        decomp_fn=bias_variance_mse,
    )

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)
    assert avg_var == net_var


def test_bias_variance_calc_parallel_0_1_no_random_state():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_compute_parallel(
        model_wrapped,
        X_train,
        y_train,
        X_test,
        y_test,
        iterations=10,
        decomp_fn=bias_variance_0_1_loss,
    )

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)
