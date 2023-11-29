import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from mvtk.bias_variance.estimators import SciKitLearnEstimatorWrapper


def create_data():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.concatenate((np.arange(3), np.arange(3)), axis=None)
    X_test = np.arange(6).reshape(3, 2)
    y_test = np.array([0, 1, 1])

    return X_train, y_train, X_test, y_test


def test_sklearn_estimator_wrapper():
    X_train, y_train, X_test, y_test = create_data()

    model = LinearRegression()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    model_test = LinearRegression()
    model_wrapped = SciKitLearnEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train)
    pred_wrapped = model_wrapped.predict(X_test)

    assert np.array_equal(pred, pred_wrapped)


def test_sklearn_estimator_wrapper_kwargs_fit():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)

    model.fit(X_train, y_train, sample_weight=[0, 0, 1, 0, 1, 0])
    pred = model.predict(X_test)

    model_test = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train, sample_weight=[0, 0, 1, 0, 1, 0])
    pred_wrapped = model_wrapped.predict(X_test)

    assert np.array_equal(pred, pred_wrapped)


def test_sklearn_estimator_wrapper_kwargs_predict():
    X_train, y_train, X_test, y_test = create_data()

    model = DecisionTreeClassifier(random_state=123)

    model.fit(X_train, y_train)
    try:
        model.predict(X_test, check_input=False)
    except ValueError as e:
        assert e.args[0] == 'X.dtype should be np.float32, got int64'
        return

    model_test = DecisionTreeClassifier(random_state=123)
    model_wrapped = SciKitLearnEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train)
    try:
        model_wrapped.predict(X_test, check_input=False)
    except ValueError as e:
        assert e.args[0] == 'X.dtype should be np.float32, got int64'
        return

    assert False
