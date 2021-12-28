from typing import Union

import numpy as np


def linear(w, b, x) -> float:
    return w * x + b


def dw_linear(w, b, x) -> float:
    return 0.0


def db_linear(w, b, x) -> float:
    return 0.0


def dx_linear(w, b, x) -> float:
    return 0.0


def sigmoid(a: Union[float | int]) -> float:
    return 1.0 / (1.0 + np.exp(-a))


def model(w, b, x) -> float:
    return sigmoid(linear(w, b, x)) * 20.0


def dw_model(w, b, x) -> float:
    return 0.0


def db_model(w, b, x) -> float:
    return 0.0


def loss(y: float, y_prim: float):
    return np.mean((y - y_prim) ** 2)


def dw_loss(y: float, y_prim: float, w, b, x) -> float:
    return 0.0


def db_loss(y: float, y_prim: float, w, b, x) -> float:
    return 0.0


def predict(w, b, x) -> float:
    return linear(w, b, x)


def fit(x, y, epochs=100, learning_rate=0.01):
    w = 0
    b = 0

    for epoch in range(epochs):
        y_prim = model(w, b, x)
        _loss = loss(y, y_prim)

        _dw_loss = dw_loss(y, y_prim, w, b, x)
        _db_loss = dw_loss(y, y_prim, w, b, x)

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
