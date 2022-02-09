import random

import numpy as np


def linear(w: float, b: float, x: np.array) -> np.array:
    return w * x + b


def dw_linear(w: float, b: float, x: np.array) -> np.array:
    return x


def db_linear(w: float, b: float, x: np.array) -> float:
    return 1.0


def dx_linear(w: float, b: float, x: np.array) -> float:
    return w


def leaky_relu(x: np.array, slope: float) -> np.array:
    return (x > 0) * x + (x <= 0) * x * slope


def dx_leaky_relu(x: np.array, slope: float) -> np.array:
    y = np.ones_like(x)
    y[y < 0] = slope

    return y


def tanh(x: np.array) -> np.array:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def dx_tanh(x: np.array) -> np.array:
    return 1 - (np.exp(x) - np.exp(-x)) ** 2 / (np.exp(x) + np.exp(-x)) ** 2


def model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> np.array:
    x_hidden = tanh(linear(w1, b1, x))

    return leaky_relu(linear(w2, b2, x_hidden), slope)


def dw1_model(w: float, b: float, x: np.array, slope: float) -> float:
    return dx_leaky_relu(x, slope) * dx_tanh(x)


def dw2_model(w: float, b: float, x: np.array, slope: float) -> float:
    return dx_leaky_relu(x, slope) * dx_tanh(x)


def db1_model(w: float, b: float, x: np.array, slope: float) -> float:
    return dx_leaky_relu(x, slope) * dx_tanh(x)


def db2_model(w: float, b: float, x: np.array, slope: float) -> float:
    return dx_leaky_relu(x, slope) * dx_tanh(x)


def loss(y: np.array, y_predicted: np.array) -> np.array:
    """
    That's a MAE cost function, we need to minimize its result.
    """
    return np.mean(np.abs(y - y_predicted))


def dw1_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array) -> np.array:
    return None


def db1_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array) -> np.array:
    return None


def dw2_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array) -> np.array:
    return None


def db2_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array) -> np.array:
    return None


def predict(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> np.array:
    return model(w1, b1, w2, b2, x, slope)


def fit(_model, x: np.array, y: np.array, epochs=1000000, learning_rate: float = 0.001, slope: float = 0.01):
    w1 = np.float64(random.random())
    b1 = np.float64(random.random())
    w2 = np.float64(random.random())
    b2 = np.float64(random.random())
    _loss = None

    for epoch in range(epochs):
        # predicted prices
        y_predicted = _model(w1, b1, w2, b2, x, slope)

        # estimate the loss (MAE) between real prices and predicted
        _loss = loss(y, y_predicted)

        _dw1_loss = dw1_loss(y, y_predicted, w1, b1, w2, b2, x)
        _db1_loss = db1_loss(y, y_predicted, w1, b1, w2, b2, x)
        _dw2_loss = dw2_loss(y, y_predicted, w1, b1, w2, b2, x)
        _db2_loss = db2_loss(y, y_predicted, w1, b1, w2, b2, x)

        w1 -= learning_rate * _dw1_loss
        b1 -= learning_rate * _db1_loss
        w2 -= learning_rate * _dw2_loss
        b2 -= learning_rate * _db2_loss

    print(f'{w1=} {b1=} {w2=} {b2=} {_loss=}')

    return w1, b1, w2, b2


def main():
    floors = np.array([1, 2, 3, 4], dtype='float64')
    prices = np.array([0.7, 1.5, 4.5, 9.5], dtype='float64')
    slope = 0.01
    print(f'{prices*100000=}')

    learning_rate = np.float64(0.0001)
    w1, b1, w2, b2 = fit(model, floors, prices, epochs=20000, learning_rate=learning_rate, slope=slope)

    print(f'Predicted price for 3 floor building: {predict(w1, b1, w2, b2, 3, slope) * 100000.0:.2f} EUR')
    print(f'Predicted price for 5 floor building: {predict(w1, b1, w2, b2, 5, slope) * 100000.0:.2f} EUR')


if __name__ == '__main__':
    main()
