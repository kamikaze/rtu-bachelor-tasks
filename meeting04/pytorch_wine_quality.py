import time
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torch.nn
from torch.nn import Module, Sequential, Linear, Tanh, LeakyReLU, Softmax, CrossEntropyLoss
from torch.optim import Adam

torch.set_default_dtype(torch.float64)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


class LossCrossEntropy:
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim

        return torch.mean(-y * torch.log(y_prim))

    def backward(self):
        self.y_prim.grad = -self.y.value / self.y_prim.value


def normalize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_values = np.max(values, axis=0)
    min_values = np.min(values, axis=0)

    return 2.0 * ((values - min_values) / (max_values - min_values) - 0.5), min_values, max_values


def denormalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return (values / 2.0 + 0.5) * (max_values - min_values) + min_values


class Model(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Linear(in_features=13, out_features=10, device=device),
            Tanh(),
            Linear(in_features=10, out_features=5, device=device),
            LeakyReLU(),
            Linear(in_features=5, out_features=3, device=device),
            Softmax(dim=1)
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)

        return y_prim


def main():
    plt.show()

    features, classes = sklearn.datasets.load_wine(return_X_y=True)
    features, _, _ = normalize(features)

    np.random.seed(0)
    idxes_rand = np.random.permutation(len(features))
    features = features[idxes_rand]
    classes = classes[idxes_rand]

    class_idxes = classes
    classes = np.zeros((len(classes), len(np.unique(classes))))
    classes[np.arange(len(classes)), class_idxes] = 1.0
    idx_split = int(len(features) * 0.9)
    dataset_train = (
        torch.tensor(features[:idx_split], device=device),
        torch.tensor(classes[:idx_split], device=device)
    )
    dataset_test = (
        torch.tensor(features[idx_split:], device=device),
        torch.tensor(classes[idx_split:], device=device)
    )
    np.random.seed(int(time.time()))

    epoch_count = 1000000
    learning_rate = 1e-6

    model = Model()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    losses_train = []
    losses_test = []
    accuracy_train = []
    accuracy_test = []

    # loss_fn = LossCrossEntropy()
    loss_fn = CrossEntropyLoss()

    for epoch in range(epoch_count+1):
        for dataset in (dataset_train, dataset_test):
            x, y = dataset

            # Every FloatTensor has value/data and grad props as Variable class had
            y_prim = model.forward(x)
            loss = loss_fn.forward(y, y_prim)

            if dataset is dataset_train:
                loss.backward()
                optimizer.step()

            if epoch % 1000 == 0:
                # Let's stop calculating gradients
                with torch.no_grad():
                    predicted = torch.max(y_prim.data, 1)[1]
                    expected = torch.max(y.data, 1)[1]
                    accuracy = (predicted == expected).sum() / y.size(0)

                    if dataset is dataset_train:
                        losses_train.append(loss.item())
                        accuracy_train.append(accuracy.item())
                    else:
                        losses_test.append(loss.item())
                        accuracy_test.append(accuracy.item())

        if epoch % 1000 == 0:
            print(f'{epoch=} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]}')
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title('Cross-Entropy Loss')
            plt.plot(losses_train, label='Train')
            plt.plot(losses_test, label='Test')
            plt.legend(loc='upper right')
            plt.xlabel('epoch x1000')
            plt.ylabel('loss')
            plt.subplot(2, 1, 2)
            plt.title('Accuracy')
            plt.plot(accuracy_train, label='Train')
            plt.plot(accuracy_test, label='Test')
            plt.legend(loc='lower right')
            plt.xlabel('epoch x1000')
            plt.ylabel('accuracy')
            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
