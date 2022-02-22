import time
from functools import reduce
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


class Variable:
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.normal(size=(out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Optional[Variable] = None
        self.output: Optional[Variable] = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output  # this output will be input for next function in model

    def backward(self):
        # W*x + b / d b = 0 + b^{1-1} = 1
        # d_b = 1 * chain_rule_of_prev_d_func
        self.b.grad = 1 * self.output.grad

        # d_W = x * chain_rule_of_prev_d_func
        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        # d_x = W * chain_rule_of_prev_d_func
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LayerReLU:
    def __init__(self):
        self.x: Optional[Variable] = None
        self.output: Optional[Variable] = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable((x.value >= 0) * x.value)
        # [1, -1, -1] >= 0
        # [True, False, False] * 1.0 => [1, 0, 0]
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad


class LayerSoftmax:
    def __init__(self):
        self.x: Optional[Variable] = None
        self.output: Optional[Variable] = None

    def forward(self, x: Variable):
        self.x = x
        np_x = np.copy(x.value)
        # numerical stability for large values
        np_x -= np.max(np_x, axis=1, keepdims=True)
        self.output = Variable(
            (np.exp(np_x + 1e-8)) / np.sum(np.exp(np_x), axis=1, keepdims=True)
        )
        return self.output

    def backward(self):
        a = self.output.value
        jacobian = np.zeros((len(a), 3, 3))

        for i in range(3):
            for j in range(3):
                if i == j:
                    # identity part of matrix / trace
                    jacobian[:, i, j] = a[:, i] * (1 - a[:, j])
                else:
                    jacobian[:, i, j] = -a[:, i] * a[:, j]

        # jacobian shape = (B, 3, 3)
        # output.grad shape = (B, 3)
        # output.grad shape with expand_dims = (B, 3, 1), [1 2 3] => [[1] [2] [3]]
        # self.x.grad shape = (B, 3, 1) => squeze (B, 3)
        self.x.grad = np.matmul(jacobian, np.expand_dims(self.output.grad, axis=2)).squeeze()


class LossCrossEntropy:
    def __init__(self):
        self.y: Optional[Variable] = None
        self.y_prim: Optional[Variable] = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim

        return np.mean(-y.value * np.log(y_prim.value))

    def backward(self):
        self.y_prim.grad = -self.y.value / self.y_prim.value


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=13, out_features=10),  # W_1*x + b < dW , db
            LayerReLU(),
            LayerLinear(in_features=10, out_features=5),  # W_2*x + b
            LayerReLU(),
            LayerLinear(in_features=5, out_features=3),  # W_3*x + b
            LayerSoftmax()
        ]

    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []

        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)

        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W (8, 4)            dW (16, 8, 4)  => (Batch, InFeatures, OutFeatures)
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate  # W = W - dW * alpha


def main():
    epoch_count = 1_000_000
    learning_rate = 1e-2
    features, classes = sklearn.datasets.load_wine(return_X_y=True)
    batch_size = len(features)

    np.random.seed(0)
    idxes_rand = np.random.permutation(len(features))
    features = features[idxes_rand]
    classes = classes[idxes_rand]

    class_idxes = classes
    classes = np.zeros((len(classes), len(np.unique(classes))))
    classes[np.arange(len(classes)), class_idxes] = 1.0

    idx_split = int(len(features) * 0.9)
    dataset_train = (features[:idx_split], classes[:idx_split])
    dataset_test = (features[idx_split:], classes[idx_split:])

    np.random.seed(int(time.time()))

    model = Model()
    optimizer = OptimizerSGD(
        model.parameters(),
        learning_rate=learning_rate
    )
    loss_fn = LossCrossEntropy()

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    plt.show()

    for epoch in range(epoch_count):
        for dataset in [dataset_train, dataset_test]:
            features, classes = dataset
            losses = []
            accuracies = []

            x_cnt = len(features)
            window = min(x_cnt, batch_size)

            for idx in range(0, x_cnt - window + 1, window):
                x = features[idx:idx + batch_size]
                y = classes[idx:idx + batch_size]

                y_prim = model.forward(Variable(value=x))
                loss = loss_fn.forward(Variable(value=y), y_prim)

                guess_cnt = reduce(
                    lambda cnt, values: cnt + (np.argmax(values[0]) == np.argmax(values[1])),
                    zip(y, y_prim.value),
                    0
                )
                accuracy = guess_cnt / len(y_prim.value)
                losses.append(loss)
                accuracies.append(accuracy)

                if dataset == dataset_train:
                    loss_fn.backward()
                    model.backward()
                    optimizer.step()

            if dataset == dataset_train:
                acc_train.append(np.mean(accuracies))
                loss_train.append(np.mean(losses))
            else:
                acc_test.append(np.mean(accuracies))
                loss_test.append(np.mean(losses))

        print(
            f'epoch: {epoch} '
            f'loss_train: {loss_train[-1]} '
            f'loss_test: {loss_test[-1]} '
            f'acc_train: {acc_train[-1]} '
            f'acc_test: {acc_test[-1]} '
        )

        if epoch % 1000 == 0:
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title('loss')
            plt.plot(loss_train, label='Train')
            plt.plot(loss_test, label='Test')
            plt.legend(loc='upper right')

            plt.subplot(2, 1, 2)
            plt.title('acc')
            plt.plot(acc_train, label='Train')
            plt.plot(acc_test, label='Test')
            plt.legend(loc='lower right')

            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
