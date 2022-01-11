import random
from typing import Union

import numpy as np


def linear(w: float, b: float, x: np.array) -> np.array:
    return w * x + b


def dw_linear(w: float, b: float, x: np.array) -> np.array:
    return x


def db_linear(w: float, b: float, x: np.array) -> float:
    return 1.0


def dx_linear(w: float, b: float, x: np.array) -> float:
    return w


def sigmoid(a: np.array) -> float:
    return 1.0 / (1.0 + np.exp(-a))


def da_sigmoid(a: np.array) -> float:
    # TODO
    return 0.0


def model(w: float, b: float, x: np.array) -> float:
    return linear(w, b, x)
    # return sigmoid(linear(w, b, x)) * 20.0


def dw_model(w: float, b: float, x: np.array) -> float:
    # TODO
    return 0.0


def db_model(w: float, b: float, x: np.array) -> float:
    # TODO
    return 0.0


def loss(y: np.array, y_prim: np.array) -> np.array:
    """
    That's a cost function, we need to minimize its result.
    """
    return np.mean((y - y_prim) ** 2)


def dw_loss(y: np.array, y_prim: np.array, w: float, b: float, x: np.array) -> np.array:
    # TODO
    return -2.0 / x.size * np.sum(dw_linear(w, b, x) * (y - y_prim) * x)


def db_loss(y: np.array, y_prim: np.array, w: float, b: float, x: np.array) -> np.array:
    # TODO
    return -2.0 / x.size + np.sum(db_linear(w, b, x) * (y - y_prim))


def predict(w: float, b: float, x: Union[float, int]) -> np.array:
    return linear(w, b, x)


def fit(x: np.array, y: np.array, epochs=100000, learning_rate=0.001):
    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        # predicted prices
        y_prim = model(w, b, x)

        # estimate the loss (MSE) between real prices and predicted
        _loss = loss(y, y_prim)

        _dw_loss = dw_loss(y, y_prim, w, b, x)
        _db_loss = db_loss(y, y_prim, w, b, x)

        w -= _dw_loss * learning_rate
        b -= _db_loss * learning_rate

        print(f'{y_prim=} {_loss=}')

    return w, b


def main():
    floors = np.array([1, 2, 3, 4])
    prices = np.array([0.7, 1.5, 4.5, 9.5])

    w, b = fit(floors, prices)

    print(f'Predicted price: {predict(w, b, 5) * 100000.0:.2f} EUR')


if __name__ == '__main__':
    main()
