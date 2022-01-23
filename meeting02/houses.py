from time import sleep

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()

IS_RUNNING = False


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
    # return sigmoid(linear(w, b, x)) * 20.0
    return linear(w, b, x)


def dw_model(w: float, b: float, x: np.array) -> float:
    return dw_linear(w, b, x)


def db_model(w: float, b: float, x: np.array) -> float:
    return db_linear(w, b, x)


def loss(y: np.array, y_predicted: np.array) -> np.array:
    """
    That's a cost function, we need to minimize its result.
    """
    return np.mean((y - y_predicted) ** 2)


def dw_loss(y: np.array, y_predicted: np.array, w: float, b: float, x: np.array) -> np.array:
    # w is theta_1
    # b + w * x == y_predicted
    # 2 * np.mean((b + w * x - y) * x) ==
    return 2 * np.mean((y_predicted - y) * x)


def db_loss(y: np.array, y_predicted: np.array, w: float, b: float, x: np.array) -> np.array:
    # b is theta_0
    # b + w * x == y_predicted
    # 2 * np.mean(b + w * x - y) ==
    return 2 * np.mean(y_predicted - y)


def predict(w: float, b: float, x: np.array) -> np.array:
    return model(w, b, x)


def fit(x: np.array, y: np.array, epochs=100000, learning_rate=0.001):
    w = 0.0
    b = 0.0
    _loss = None
    y_predicted = None

    for epoch in range(epochs):
        # predicted prices
        y_predicted = model(w, b, x)

        # estimate the loss (MSE) between real prices and predicted
        _loss = loss(y, y_predicted)

        _dw_loss = dw_loss(y, y_predicted, w, b, x)
        _db_loss = db_loss(y, y_predicted, w, b, x)

        w -= learning_rate * _dw_loss
        b -= learning_rate * _db_loss

    print(f'{w=} {b=} {y_predicted=} {_loss=}')

    return w, b


def on_key_press(event):
    global IS_RUNNING

    if event.key == 'escape':
        IS_RUNNING = False


def main():
    global IS_RUNNING
    floors = np.array([1, 2, 3, 4])
    prices = np.array([0.7, 1.5, 4.5, 9.5])

    w, b = fit(floors, prices)

    print(f'Predicted price: {predict(w, b, 3) * 100000.0:.2f} EUR')

    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.scatter(floors, prices)

    x = np.linspace(0.0, 10.0, 1000)
    plt.plot(x, model(w, b, x), color='r')
    plt.legend([None, f'{w=} {b=}'])
    plt.draw()

    IS_RUNNING = True

    while IS_RUNNING:
        plt.pause(0.1)
        sleep(0.1)


if __name__ == '__main__':
    main()
