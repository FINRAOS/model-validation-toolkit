import numpy as np
import tensorflow as tf

from mvtk.bias_variance.estimators import TensorFlowEstimatorWrapper


def create_data():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.concatenate((np.arange(3), np.arange(3)), axis=None)
    X_test = np.arange(6).reshape(3, 2)
    y_test = np.array([0, 1, 1])

    return X_train, y_train, X_test, y_test


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["mean_squared_error"],
    )

    return model


def predict(estimator, X, **kwargs):
    predictions = estimator.predict(X, **kwargs)
    prediction_list = []
    for prediction in predictions:
        if len(prediction) > 1:
            prediction_list.append(prediction.argmax().item())
        else:
            prediction_list.append(prediction.item())
    return prediction_list


def test_tensorflow_estimator_wrapper():
    X_train, y_train, X_test, y_test = create_data()

    tf.keras.utils.set_random_seed(123)
    model = create_model()

    model.fit(X_train, y_train)
    pred = predict(model, X_test)

    tf.keras.utils.set_random_seed(123)
    model_test = create_model()
    model_wrapped = TensorFlowEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train)
    pred_wrapped = model_wrapped.predict(X_test)

    assert np.array_equal(pred, pred_wrapped)


def test_tensorflow_estimator_wrapper_kwargs_fit():
    X_train, y_train, X_test, y_test = create_data()

    tf.keras.utils.set_random_seed(123)
    model = create_model()

    model.fit(X_train, y_train, epochs=10)
    pred = predict(model, X_test)

    tf.keras.utils.set_random_seed(123)
    model_test = create_model()
    model_wrapped = TensorFlowEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train, epochs=10)
    pred_wrapped = model_wrapped.predict(X_test)

    assert np.array_equal(pred, pred_wrapped)


def test_tensorflow_estimator_wrapper_kwargs_predict():
    X_train, y_train, X_test, y_test = create_data()

    tf.keras.utils.set_random_seed(123)
    model = create_model()

    model.fit(X_train, y_train)
    pred = predict(model, X_test, steps=10)

    tf.keras.utils.set_random_seed(123)
    model_test = create_model()
    model_wrapped = TensorFlowEstimatorWrapper(model_test)

    model_wrapped.fit(X_train, y_train)
    pred_wrapped = model_wrapped.predict(X_test, steps=10)

    assert np.array_equal(pred, pred_wrapped)
