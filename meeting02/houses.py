import random
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
# Linear
# X_MIN = -2.0
# X_MAX = 3.5
# Y_MIN = -4.0
# Y_MAX = 5.0

# Sigmoid
X_MIN = -10.0
X_MAX = 10.0
Y_MIN = -20.0
Y_MAX = 30.0

# Cubic
# X_MIN = 0.139
# X_MAX = 0.144
# Y_MIN = 0.41
# Y_MAX = 0.57
LIN_SPACE = np.linspace(0.0, 10.0, 1000)


def linear(w: float, b: float, x: np.array) -> np.array:
    return w * x + b


def cubic(w: float, b: float, x: np.array) -> np.array:
    return w * x**3 + b


def dw_linear(w: float, b: float, x: np.array) -> np.array:
    return x


def db_linear(w: float, b: float, x: np.array) -> float:
    return 1.0


def dx_linear(w: float, b: float, x: np.array) -> float:
    return w


def dx_cubic(w: float, b: float, x: np.array) -> float:
    return w * 3 * x**2


def sigmoid(a: np.array) -> float:
    return 1.0 / (1.0 + np.exp(-a))


def da_sigmoid(a: np.array) -> float:
    sigmoid_a = sigmoid(a)

    return sigmoid_a * (1 - sigmoid_a)


def model(w: float, b: float, x: np.array) -> float:
    return sigmoid(linear(w, b, x)) * 20.0
    # return linear(w, b, x)
    # return cubic(w, b, x)


def dw_model(w: float, b: float, x: np.array) -> float:
    return da_sigmoid(linear(w, b, x)) * dw_linear(w, b, x)
    # return dw_linear(w, b, x)  # same for dw_cubic


def db_model(w: float, b: float, x: np.array) -> float:
    return da_sigmoid(linear(w, b, x)) * db_linear(w, b, x)
    # return db_linear(w, b, x)  # same for db_cubic


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


def fit(_model, x: np.array, y: np.array, epochs=1000000, learning_rate=None, callback=None):
    w = np.float64(random.random())
    b = np.float64(random.random())
    _loss = None

    current_learning_rate = learning_rate_start = max(learning_rate)
    learning_rate_end = min(learning_rate)
    learning_rate_range = learning_rate_start - learning_rate_end

    if callback:
        xw = np.linspace(X_MIN, X_MAX, 50, dtype='float64')
        yb = np.linspace(Y_MIN, Y_MAX, 50, dtype='float64')
        xw, yb = np.meshgrid(xw, yb)

        z = np.array([loss(y, _model(xx, yy, x)) for xx, yy in zip(np.ravel(xw), np.ravel(yb))], dtype='float64')
        z = z.reshape(xw.shape)

    for epoch in range(epochs):
        # predicted prices
        y_predicted = _model(w, b, x)

        # estimate the loss (MSE) between real prices and predicted
        _loss = loss(y, y_predicted)

        _dw_loss = dw_loss(y, y_predicted, w, b, x)
        _db_loss = db_loss(y, y_predicted, w, b, x)

        if epoch % 100 == 0:
            current_learning_rate = learning_rate_start - learning_rate_range * (epoch / epochs)

        new_w = w - current_learning_rate * _dw_loss
        new_b = b - current_learning_rate * _db_loss

        # Checking if we are able to improve further. There is a big chance that loss will be not 0 but computer
        # will not be able to change w or b due to precision limitations. So we will check if change happened instead.
        if new_w == w and new_b == b:
            print(f'{epoch=}: We are at minimum')
            break

        w = new_w
        b = new_b

        if callback:
            # if epoch < 50 or epoch % 10 == 0:
            if epoch % 20 == 0:
                callback(x, y, w, b, current_learning_rate, _loss, xw, yb, z)

            if epoch == 0:
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                callback(x, y, w, b, current_learning_rate, _loss, xw, yb, z)

            # plt.savefig(f'plot_{epoch:0>8}.png')

    if callback:
        callback(x, y, w, b, current_learning_rate, _loss, xw, yb, z)

    print(f'{w=} {b=} {_loss=}')

    return w, b


def on_key_press(event):
    global IS_RUNNING

    if event.key == 'escape':
        IS_RUNNING = False


def update_plot(fig, x: np.array, y: np.array, w: float, b: float, learning_rate: float, _loss: float,
                xw, yb, z):
    plt.clf()
    fig.suptitle(f'{w=} {b=} loss={_loss} {learning_rate=}')

    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(x, y)
    # ax.legend([None, 'model'])
    ax.plot(LIN_SPACE, model(w, b, LIN_SPACE), color='r')
    ax.set_title('Model')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.view_init(None, -85)

    surf = ax.plot_surface(xw, yb, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False, alpha=0.6)
    ax.scatter(w, b, _loss, color='black')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Loss')

    plt.draw()

    plt.pause(0.001)


def main():
    global IS_RUNNING
    floors = np.array([1, 2, 3, 4], dtype='float64')
    prices = np.array([0.7, 1.5, 4.5, 9.5], dtype='float64')
    print(f'{prices*100000=}')

    fig = plt.figure(figsize=plt.figaspect(1.))
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

    callback = partial(update_plot, fig)
    learning_rate = [np.float64(0.005), np.float64(0.0001)]
    w, b = fit(model, floors, prices, epochs=10000, learning_rate=learning_rate, callback=callback)

    print(f'Predicted price for 3 floor building: {predict(w, b, 3) * 100000.0:.2f} EUR')
    print(f'Predicted price for 5 floor building: {predict(w, b, 5) * 100000.0:.2f} EUR')

    IS_RUNNING = True

    while IS_RUNNING:
        plt.pause(0.1)
        sleep(0.1)


if __name__ == '__main__':
    main()
