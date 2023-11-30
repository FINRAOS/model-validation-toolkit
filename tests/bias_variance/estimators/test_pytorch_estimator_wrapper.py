import numpy as np
import torch
from torch import nn

from mvtk.bias_variance.estimators import PyTorchEstimatorWrapper


class ModelPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def create_data():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.concatenate((np.arange(3), np.arange(3)), axis=None)
    X_test = np.arange(6).reshape(3, 2)
    y_test = np.array([0, 1, 1])

    return X_train, y_train, X_test, y_test


def create_model():
    model_pytorch = ModelPyTorch()
    optimizer = torch.optim.Adam(model_pytorch.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    return model_pytorch, optimizer, loss_fn


def optimizer_gen(x):
    return torch.optim.Adam(x.parameters(), lr=0.001)


def reset_parameters(x):
    if hasattr(x, "reset_parameters"):
        x.reset_parameters()


def fit(estimator, optimizer, loss_fn, X, y, epochs=10, batch_size=None):
    for i in range(epochs):
        if batch_size is None:
            batch_size = len(y)
        for j in range(0, len(y), batch_size):
            batch_start = j
            batch_end = j + batch_size
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            prediction = estimator(X_batch)
            loss = loss_fn(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def custom_fit(self, X, y, epochs=10, batch_size=None):
    for i in range(epochs):
        if batch_size is None:
            batch_size = len(y)
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


def predict(estimator, X, custom_test=False):
    if custom_test:
        return [1, 0, 1]

    prediction_list = []
    with torch.no_grad():
        for value in X:
            prediction = estimator(value)
            if len(prediction) > 1:
                prediction_list.append(prediction.argmax().item())
            else:
                prediction_list.append(prediction.item())
    return prediction_list


def custom_predict(estimator, X):
    return [1, 0, 1]


def test_pytorch_estimator_wrapper():
    torch.use_deterministic_algorithms(True)

    X_train, y_train, X_test, y_test = create_data()

    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)

    torch.manual_seed(123)
    model, optimizer, loss_fn = create_model()

    model.apply(reset_parameters)
    fit(model, optimizer, loss_fn, X_train_torch, y_train_torch, epochs=100)
    pred = predict(model, X_test_torch)

    torch.manual_seed(123)
    model_test, optimizer_test, loss_fn_test = create_model()
    model_wrapped = PyTorchEstimatorWrapper(model_test, optimizer_gen, loss_fn_test)

    model_wrapped.fit(X_train_torch, y_train_torch)
    pred_wrapped = model_wrapped.predict(X_test_torch)

    assert np.array_equal(pred, pred_wrapped)


def test_pytorch_estimator_wrapper_kwargs_fit():
    torch.use_deterministic_algorithms(True)

    X_train, y_train, X_test, y_test = create_data()

    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)

    torch.manual_seed(123)
    model, optimizer, loss_fn = create_model()

    model.apply(reset_parameters)
    fit(model, optimizer, loss_fn, X_train_torch, y_train_torch, epochs=5)
    pred = predict(model, X_test_torch)

    torch.manual_seed(123)
    model_test, optimizer_test, loss_fn_test = create_model()
    model_wrapped = PyTorchEstimatorWrapper(model_test, optimizer_gen, loss_fn_test)

    model_wrapped.fit(X_train_torch, y_train_torch, epochs=5)
    pred_wrapped = model_wrapped.predict(X_test_torch)

    assert np.array_equal(pred, pred_wrapped)


def test_pytorch_estimator_wrapper_custom_fit():
    torch.use_deterministic_algorithms(True)

    X_train, y_train, X_test, y_test = create_data()

    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)

    torch.manual_seed(123)
    model, optimizer, loss_fn = create_model()

    model.apply(reset_parameters)
    fit(model, optimizer, loss_fn, X_train_torch, y_train_torch, epochs=10)
    pred = predict(model, X_test_torch)

    torch.manual_seed(123)
    model_test, optimizer_test, loss_fn_test = create_model()
    model_wrapped = PyTorchEstimatorWrapper(model_test, optimizer_gen, loss_fn_test,
                                            fit_fn=custom_fit)

    model_wrapped.fit(X_train_torch, y_train_torch)
    pred_wrapped = model_wrapped.predict(X_test_torch)

    assert np.array_equal(pred, pred_wrapped)


def test_pytorch_estimator_wrapper_custom_predict():
    torch.use_deterministic_algorithms(True)

    X_train, y_train, X_test, y_test = create_data()

    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)

    torch.manual_seed(123)
    model, optimizer, loss_fn = create_model()

    model.apply(reset_parameters)
    fit(model, optimizer, loss_fn, X_train_torch, y_train_torch, epochs=100)
    pred = predict(model, X_test_torch, custom_test=True)

    torch.manual_seed(123)
    model_test, optimizer_test, loss_fn_test = create_model()
    model_wrapped = PyTorchEstimatorWrapper(model_test, optimizer_gen, loss_fn_test,
                                            predict_fn=custom_predict)

    model_wrapped.fit(X_train_torch, y_train_torch)
    pred_wrapped = model_wrapped.predict(X_test_torch)

    assert np.array_equal(pred, pred_wrapped)
