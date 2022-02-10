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


def leaky_relu(k: np.array, slope: float) -> np.array:
    return (k > 0) * k + (k <= 0) * k * slope


def dk_leaky_relu(k: np.array, slope: float) -> np.array:
    _k = np.ones_like(k)
    _k[_k < 0] = slope

    return _k


def tanh(m: np.array) -> np.array:
    return (np.exp(m) - np.exp(-m)) / (np.exp(m) + np.exp(-m))


def dm_tanh(m: np.array) -> np.array:
    return 1 - (np.exp(m) - np.exp(-m)) ** 2 / (np.exp(m) + np.exp(-m)) ** 2


def model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> np.array:
    _m = linear(w1, b1, x)
    _l = tanh(_m)
    _k = linear(w2, b2, _l)

    return leaky_relu(_k, slope)


def dw2_model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> float:
    _m = linear(w1, b1, x)
    _l = tanh(_m)
    _k = linear(w2, b2, _l)

    return dk_leaky_relu(_k, slope) * _l


def db2_model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> float:
    _m = linear(w1, b1, x)
    _l = tanh(_m)
    _k = linear(w2, b2, _l)

    return dk_leaky_relu(_k, slope) * 1


def dw1_model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> float:
    _m = linear(w1, b1, x)
    _l = tanh(_m)
    _k = linear(w2, b2, _l)

    return dk_leaky_relu(_k, slope) * dx_linear(w2, b2, _l) * dm_tanh(_m) * x


def db1_model(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> float:
    _m = linear(w1, b1, x)
    _l = tanh(_m)
    _k = linear(w2, b2, _l)

    return dk_leaky_relu(_k, slope) * dx_linear(w2, b2, _l) * dm_tanh(_m) * 1


def loss(y: np.array, y_predicted: np.array) -> np.array:
    """
    That's a MAE cost function, we need to minimize its result.
    """
    return np.mean(np.abs(y - y_predicted))


def dy_predicted_loss(y: np.array, y_predicted: np.array) -> np.array:
    diff = y - y_predicted

    return -(diff / np.abs(diff))


def dw2_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array,
             slope: float) -> np.array:
    return np.mean(dy_predicted_loss(y, y_predicted) * dw2_model(w1, b1, w2, b2, x, slope))


def db2_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array,
             slope: float) -> np.array:
    return np.mean(dy_predicted_loss(y, y_predicted) * db2_model(w1, b1, w2, b2, x, slope))


def dw1_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array,
             slope: float) -> np.array:
    return np.mean(dy_predicted_loss(y, y_predicted) * dw1_model(w1, b1, w2, b2, x, slope))


def db1_loss(y: np.array, y_predicted: np.array, w1: float, b1: float, w2: float, b2: float, x: np.array,
             slope: float) -> np.array:
    return np.mean(dy_predicted_loss(y, y_predicted) * db1_model(w1, b1, w2, b2, x, slope))


def predict(w1: float, b1: float, w2: float, b2: float, x: np.array, slope: float) -> np.array:
    return model(w1, b1, w2, b2, x, slope)


def fit(_model, x: np.array, y: np.array, epochs=1000000, learning_rate: float = 0.001, slope: float = 0.01):
    best_params = [None, None, None, None]
    w1 = np.float64(random.random())
    b1 = np.float64(random.random())
    w2 = np.float64(random.random())
    b2 = np.float64(random.random())
    _loss = None
    _best_loss = None

    for epoch in range(epochs):
        # predicted prices
        y_predicted = _model(w1, b1, w2, b2, x, slope)

        # estimate the loss (MAE) between real prices and predicted
        _loss = loss(y, y_predicted)

        if _best_loss is None or _loss < _best_loss:
            best_params[:] = w1, b1, w2, b2

        if epoch % 1000 == 0:
            print(f'{epoch=} {_loss=}')

        _dw1_loss = dw1_loss(y, y_predicted, w1, b1, w2, b2, x, slope)
        _db1_loss = db1_loss(y, y_predicted, w1, b1, w2, b2, x, slope)
        _dw2_loss = dw2_loss(y, y_predicted, w1, b1, w2, b2, x, slope)
        _db2_loss = db2_loss(y, y_predicted, w1, b1, w2, b2, x, slope)

        w1 -= learning_rate * _dw1_loss
        b1 -= learning_rate * _db1_loss
        w2 -= learning_rate * _dw2_loss
        b2 -= learning_rate * _db2_loss

    print(f'w1={best_params[0]} b1={best_params[1]} w2={best_params[2]} b2={best_params[3]} loss={_best_loss}')

    return best_params


def main():
    floors = np.array([1, 2, 3, 4], dtype='float64')
    prices = np.array([0.7, 1.5, 4.5, 9.5], dtype='float64')
    slope = 0.000001
    print(f'{prices*100000=}')

    learning_rate = np.float64(0.00005)
    w1, b1, w2, b2 = fit(model, floors, prices, epochs=2000000, learning_rate=learning_rate, slope=slope)

    print(f'Predicted price for 2 floor building: {predict(w1, b1, w2, b2, 2, slope) * 100000.0:.2f} EUR')
    print(f'Predicted price for 3 floor building: {predict(w1, b1, w2, b2, 3, slope) * 100000.0:.2f} EUR')
    print(f'Predicted price for 5 floor building: {predict(w1, b1, w2, b2, 5, slope) * 100000.0:.2f} EUR')


if __name__ == '__main__':
    main()
