from functools import partial
from time import sleep

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()

IS_RUNNING = False
X_MIN = -15.0
X_MAX = 35.0
Y_MIN = -15.0
Y_MAX = 50.0
LIN_SPACE = np.linspace(0.0, 10.0, 1000)


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
    return sigmoid(linear(w, b, x)) * 20.0
    # return linear(w, b, x)


def dw_model(w: float, b: float, x: np.array) -> float:
    # return da_sigmoid(linear(w, b, x))
    return dw_linear(w, b, x)


def db_model(w: float, b: float, x: np.array) -> float:
    # return da_sigmoid(linear(w, b, x))
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


def fit(_model, x: np.array, y: np.array, epochs=1000000, learning_rate=0.001, callback=None):
    w = 30.0
    b = 50.0
    _loss = None
    y_predicted = None

    if callback:
        xw = np.linspace(X_MIN, X_MAX, 200)
        yb = np.linspace(Y_MIN, Y_MAX, 200)
        xw, yb = np.meshgrid(xw, yb)

        z = np.array([loss(y, _model(xx, yy, x)) for xx, yy in zip(np.ravel(xw), np.ravel(yb))])
        z = z.reshape(xw.shape)

    for epoch in range(epochs):
        # predicted prices
        y_predicted = _model(w, b, x)

        # estimate the loss (MSE) between real prices and predicted
        _loss = loss(y, y_predicted)

        _dw_loss = dw_loss(y, y_predicted, w, b, x)
        _db_loss = db_loss(y, y_predicted, w, b, x)

        w -= learning_rate * _dw_loss
        b -= learning_rate * _db_loss

        if epoch % 100 == 0:
            if callback:
                speed = max(learning_rate * _dw_loss, learning_rate * _db_loss)
                callback(x, y, w, b, _loss, xw, yb, z, speed)

    if callback:
        callback(x, y, w, b, _loss, xw, yb, z)

    print(f'{w=} {b=} {y_predicted=} {_loss=}')

    return w, b


def on_key_press(event):
    global IS_RUNNING

    if event.key == 'escape':
        IS_RUNNING = False


def update_plot(fig, x: np.array, y: np.array, w: float, b: float, _loss: float,
                xw, yb, z, speed: float):
    plt.clf()
    fig.suptitle(f'{w=} {b=} loss={_loss}')

    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(x, y)
    # ax.legend([None, 'model'])
    ax.plot(LIN_SPACE, model(w, b, LIN_SPACE), color='r')
    ax.set_title('Model')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(None, -85)

    ax.scatter(w, b, _loss, color='black')

    surf = ax.plot_surface(xw, yb, z, rstride=4, cstride=4, cmap=cm.coolwarm, linewidth=1, antialiased=False, alpha=0.5)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Loss')

    plt.draw()

    plt.pause(0.001)


def main():
    global IS_RUNNING
    floors = np.array([1, 2, 3, 4])
    prices = np.array([0.7, 1.5, 4.5, 9.5])

    fig = plt.figure(figsize=plt.figaspect(1.))
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

    callback = partial(update_plot, fig)
    w, b = fit(model, floors, prices, learning_rate=0.0005, callback=callback)

    print(f'Predicted price: {predict(w, b, 3) * 100000.0:.2f} EUR')

    plt.savefig('plot.png')

    IS_RUNNING = True

    while IS_RUNNING:
        plt.pause(0.1)
        sleep(0.1)


if __name__ == '__main__':
    main()
