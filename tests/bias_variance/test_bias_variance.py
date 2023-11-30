import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from mvtk.bias_variance import (bias_variance_compute, bias_variance_mse,
                                bias_variance_0_1_loss, get_values,
                                train_and_predict, bootstrap_train_and_predict)
from mvtk.bias_variance.estimators import SciKitLearnEstimatorWrapper


def create_data():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.concatenate((np.arange(3), np.arange(3)), axis=None)
    X_test = np.arange(6).reshape(3, 2)
    y_test = np.array([0, 1, 1])

    return X_train, y_train, X_test, y_test


def test_get_values():
    a = [1, 2]
    b = [3, 4]
    c = [1, 3]
    d = [2, 4]
    df = pd.DataFrame(data={'col_a': a, 'col_b': b})

    df_values = get_values(df)
    np_array = np.asarray([c, d])

    assert isinstance(df_values, np.ndarray)
    assert np.array_equal(df_values, np_array)


def test_train_and_predict_default():
    X_train, y_train, X_test, y_test = create_data()

    model = LinearRegression()
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    predictions = train_and_predict(model_wrapped, X_train, y_train, X_test)

    expected = np.array([0.4285714285714284, 0.657142857142857, 0.8857142857142857])

    assert np.array_equal(predictions, expected)


def test_train_and_predict_prepare():
    X_train, y_train, X_test, y_test = create_data()

    model = LinearRegression()
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    predictions = train_and_predict(model_wrapped, X_train, y_train, X_test,
                                    prepare_X=lambda x: x + 1,
                                    prepare_y_train=lambda x: x + 1)

    expected = np.array([1.314285714285714, 1.5428571428571427, 1.7714285714285714])

    assert np.array_equal(predictions, expected)


def test_train_and_predict_kwargs_fit():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    predictions = train_and_predict(model_wrapped, X_train, y_train, X_test,
                                    fit_kwargs={'sample_weight': [0, 0, 1, 0, 1, 0]})

    expected = np.array([2, 2, 2])

    assert np.array_equal(predictions, expected)


def test_train_and_predict_kwargs_predict():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    train_and_predict(model_wrapped, X_train, y_train, X_test)

    try:
        train_and_predict(model_wrapped, X_train, y_train, X_test,
                          predict_kwargs={'check_input': False})
    except ValueError as e:
        assert e.args[0] == 'X.dtype should be np.float32, got int64'
        return

    assert False


def test_bootstrap_train_and_predict_default():
    X_train, y_train, X_test, y_test = create_data()

    model = LinearRegression()
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    predictions = bootstrap_train_and_predict(model_wrapped, X_train, y_train, X_test,
                                              random_state=321)

    expected = np.array([0.7142857142857142, 0.8571428571428571, 1.0])

    assert np.array_equal(predictions, expected)


def test_bootstrap_train_and_predict_kwargs_fit():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    predictions = bootstrap_train_and_predict(model_wrapped, X_train, y_train, X_test,
                                              random_state=321,
                                              fit_kwargs={'sample_weight':
                                                          [0, 0, 1, 0, 1, 0]})

    expected = np.array([0, 0, 0])

    assert np.array_equal(predictions, expected)


def test_bootstrap_train_and_predict_kwargs_predict():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    bootstrap_train_and_predict(model_wrapped, X_train, y_train, X_test,
                                random_state=321)

    try:
        bootstrap_train_and_predict(model_wrapped, X_train, y_train, X_test,
                                    random_state=321,
                                    predict_kwargs={'check_input': False})
    except ValueError as e:
        assert e.args[0] == 'X.dtype should be np.float32, got int64'
        return

    assert False


def test_bias_variance_compute_mse():
    X_train, y_train, X_test, y_test = create_data()

    model = LinearRegression()
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = (
        bias_variance_compute(model_wrapped, X_train, y_train, X_test, y_test,
                              iterations=10, random_state=123,
                              decomp_fn=bias_variance_mse))

    assert avg_loss == np.float64(1.1661215949979167)
    assert avg_bias == np.float64(0.11952943334828559)
    assert avg_var == np.float64(1.0465921616496312)
    assert net_var == np.float64(1.0465921616496312)

    assert avg_loss == avg_bias + net_var
    assert avg_var == net_var


def test_bias_variance_compute_0_1():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model)

    avg_loss, avg_bias, avg_var, net_var = (
        bias_variance_compute(model_wrapped, X_train, y_train, X_test, y_test,
                              iterations=10, random_state=123,
                              decomp_fn=bias_variance_0_1_loss))

    assert avg_loss == np.float64(0.4666666666666666)
    assert avg_bias == np.float64(0.3333333333333333)
    assert avg_var == np.float64(0.3666666666666667)
    assert net_var == np.float64(0.1333333333333333)

    assert avg_loss == avg_bias + net_var


def test_bias_variance_mse_no_loss():
    predictions = np.zeros((3, 5))
    y_test = np.zeros(5)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_mse(predictions, y_test)

    assert avg_loss == np.float64(0.0)
    assert avg_bias == np.float64(0.0)
    assert avg_var == np.float64(0.0)
    assert net_var == np.float64(0.0)

    assert avg_loss == avg_bias + net_var
    assert avg_var == net_var


def test_bias_variance_mse():
    predictions = np.zeros((3, 5))
    predictions[0] += 0.5
    y_test = np.zeros(5)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_mse(predictions, y_test)

    assert avg_loss == np.float64(0.08333333333333333)
    assert avg_bias == np.float64(0.02777777777777778)
    assert avg_var == np.float64(0.05555555555555556)
    assert net_var == np.float64(0.05555555555555556)

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)
    assert avg_var == net_var


def test_bias_variance_0_1_loss_no_loss():
    predictions = np.zeros((3, 5))
    y_test = np.zeros(5)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_0_1_loss(predictions, y_test)

    assert avg_loss == np.float64(0.0)
    assert avg_bias == np.float64(0.0)
    assert avg_var == np.float64(0.0)
    assert net_var == np.float64(0.0)

    assert avg_loss == avg_bias + net_var


def test_bias_variance_0_1_loss_no_bias():
    predictions = np.zeros((3, 5))
    predictions[0] += 1
    y_test = np.zeros(5)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_0_1_loss(predictions, y_test)

    assert avg_loss == np.float64(0.3333333333333333)
    assert avg_bias == np.float64(0.0)
    assert avg_var == np.float64(0.3333333333333333)
    assert net_var == np.float64(0.3333333333333333)

    assert avg_loss == avg_bias + net_var


def test_bias_variance_0_1_loss_var_diff():
    predictions = np.zeros((3, 5))
    predictions[0] += 1
    predictions[1][0] += 1
    y_test = np.zeros(5)
    y_test[1] += 1

    avg_loss, avg_bias, avg_var, net_var = bias_variance_0_1_loss(predictions, y_test)

    assert avg_loss == np.float64(0.4666666666666666)
    assert avg_bias == np.float64(0.4)
    assert avg_var == np.float64(0.3333333333333333)
    assert net_var == np.float64(0.06666666666666668)

    assert np.round(avg_loss, decimals=12) == np.round(avg_bias + net_var, decimals=12)


def test_bias_variance_0_1_loss_div_by_0():
    predictions = np.ones((3, 5))
    y_test = np.zeros(5)

    avg_loss, avg_bias, avg_var, net_var = bias_variance_0_1_loss(predictions, y_test)

    assert avg_loss == np.float64(1.0)
    assert avg_bias == np.float64(1.0)
    assert avg_var == np.float64(0.0)
    assert net_var == np.float64(0.0)

    assert avg_loss == avg_bias + net_var
