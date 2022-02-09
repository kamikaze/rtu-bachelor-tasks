import time
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


class Variable:
    def __init__(self, value: np.ndarray):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)

    def __repr__(self):
        return f'{self.value}'


class LinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.w = Variable(value=np.random.random((out_features, in_features)))
        self.b = Variable(value=np.zeros((out_features,)))
        self.x: Optional[Variable] = None
        self.output: Optional[Variable] = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            np.matmul(self.w.value, x.value.transpose()).transpose() + self.b.value
        )

        return self.output

    def backward(self):
        self.b.grad = 1 * self.output.grad
        self.w.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )
        self.x.grad = np.matmul(
            self.w.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class TanhLayer:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        x_exp = np.exp(x.value)
        minus_x_exp = np.exp(-x.value)
        self.output = Variable(
            (x_exp - minus_x_exp) / (x_exp + minus_x_exp)
        )
        return self.output

    def backward(self):
        x_exp = np.exp(self.x.value)
        minus_x_exp = np.exp(-self.x.value)
        self.x.grad = (1 - (x_exp - minus_x_exp) ** 2 / (x_exp + minus_x_exp) ** 2) * self.output.grad


class SigmoidLayer:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            1.0 / (1.0 + np.exp(-x.value))
        )
        return self.output

    def backward(self):
        self.x.grad = -1.0 / (1.0 + np.exp(-self.x.value)) ** 2 * self.output.grad


class ReLULayer:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            (x.value >= 0) * x.value
        )
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad


class LeakyReLULayer:
    def __init__(self, slope: float = 0.0000001):
        self.x = None
        self.output = None
        self.slope = slope

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            (x.value > 0) * x.value + (x.value <= 0) * x.value * self.slope
        )

        return self.output

    def backward(self):
        self.x.grad = (self.x.value > 0) * self.output.grad + (self.x.value <= 0) * self.output.grad * self.slope


class MAELoss:
    def __init__(self):
        self.y: Optional[Variable] = None
        self.y_prim: Optional[Variable] = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        return np.mean(np.abs(y.value - y_prim.value))

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value) / np.abs(self.y.value - self.y_prim.value)


class MSELoss:
    def __init__(self):
        self.y: Optional[Variable] = None
        self.y_prim: Optional[Variable] = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        return np.mean((y.value - y_prim.value) ** 2)

    def backward(self):
        self.y_prim.grad = 2.0 * (self.y_prim.value - self.y.value)


class Model:
    def __init__(self):
        self.layers = [
            LinearLayer(in_features=8, out_features=4),
            SigmoidLayer(),
            LinearLayer(in_features=4, out_features=4),
            SigmoidLayer(),
            LinearLayer(in_features=4, out_features=1)
        ]

    def forward(self, x: Variable) -> Variable:
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self) -> Sequence[Variable]:
        variables = []

        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                variables.append(layer.w)
                variables.append(layer.b)

        return variables


class SGDOptimizer:
    def __init__(self, parameters: Sequence[Variable], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate


def normalize(values: np.ndarray) -> np.ndarray:
    min_values = np.min(values, axis=0)

    return 2 * ((values - min_values) / (np.max(values, axis=0) - min_values) - 0.5)


def main():
    plt.show()

    data_x, data_y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    data_y = np.expand_dims(data_y, axis=1)

    data_x = normalize(data_x)
    data_y = normalize(data_y)

    np.random.seed(0)
    idx_rand = np.random.permutation(len(data_x))
    data_x = data_x[idx_rand]
    data_y = data_y[idx_rand]

    idx_split = int(len(data_x) * 0.8)
    dataset_train = (data_x[:idx_split], data_y[:idx_split])
    dataset_test = (data_x[idx_split:], data_y[idx_split:])
    np.random.seed(int(time.time()))

    epoch_count = 300
    learning_rate = 0.005
    batch_size = 32

    model = Model()
    optimizer = SGDOptimizer(model.parameters(), learning_rate)
    loss_fn = MSELoss()
    losses_train = []
    losses_test = []

    for epoch in range(epoch_count):
        for dataset in (dataset_train, dataset_test):
            losses = []

            for idx in range(0, len(data_x) - batch_size, batch_size):
                x = data_x[idx:idx+batch_size]
                y = data_y[idx:idx+batch_size]

                y_prim = model.forward(Variable(x))
                loss = loss_fn.forward(Variable(y), y_prim)

                losses.append(loss)

                if dataset is dataset_train:
                    loss_fn.backward()
                    model.backward()
                    optimizer.step()

            mean_loss = np.mean(losses)

            if dataset is dataset_train:
                losses_train.append(mean_loss)
            else:
                losses_test.append(mean_loss)

        print(f'{epoch=} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]}')

        if epoch % 10 == 0:
            plt.clf()
            plt.plot(losses_train)
            plt.plot(losses_test)
            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
