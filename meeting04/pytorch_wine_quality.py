import time
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torch.nn
from torch.nn import Module, Sequential, Linear, Tanh, LeakyReLU
from torch.optim import Adam

torch.set_default_dtype(torch.float64)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


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
            Linear(in_features=5, out_features=1, device=device)
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)

        return y_prim


def main():
    plt.show()

    data_x, data_y = sklearn.datasets.load_wine(return_X_y=True)
    data_y = np.expand_dims(data_y, axis=1)

    data_x, _, _ = normalize(data_x)
    # TODO convert class number into array 0 -> [1 0 0] etc.
    data_y, _, _ = normalize(data_y)

    np.random.seed(0)
    idx_rand = np.random.permutation(len(data_x))
    data_x = data_x[idx_rand]
    data_y = data_y[idx_rand]

    idx_split = int(len(data_x) * 0.8)
    dataset_train = (
        torch.tensor(data_x[:idx_split], device=device),
        torch.tensor(data_y[:idx_split], device=device)
    )
    dataset_test = (
        torch.tensor(data_x[idx_split:], device=device),
        torch.tensor(data_y[idx_split:], device=device)
    )
    np.random.seed(int(time.time()))

    epoch_count = 1000000
    learning_rate = 1e-6

    model = Model()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    losses_train = []
    losses_test = []
    nrmse_train = []
    nrmse_test = []

    y_max_train = torch.max(dataset_train[1])
    y_min_train = torch.min(dataset_train[1])
    y_max_test = torch.max(dataset_test[1])
    y_min_test = torch.min(dataset_test[1])

    for epoch in range(epoch_count+1):
        for dataset in (dataset_train, dataset_test):
            x, y = dataset

            # Every FloatTensor has value/data and grad props as Variable class had
            y_prim = model.forward(x)
            loss = torch.mean((y - y_prim) ** 2)

            if dataset is dataset_train:
                scaler = 1.0 / (y_max_train - y_min_train)
                loss.backward()
                optimizer.step()
            else:
                scaler = 1.0 / (y_max_test - y_min_test)

            if epoch % 1000 == 0:
                # Let's stop calculating gradients for loss
                nrmse = torch.mean(scaler * torch.sqrt(torch.mean(loss.detach())))
                # same as nrmse = scaler * torch.sqrt(torch.mean((y.detach() - y_prim.detach()) ** 2))

                if dataset is dataset_train:
                    losses_train.append(loss.item())
                    nrmse_train.append(nrmse.item())
                else:
                    losses_test.append(loss.item())
                    nrmse_test.append(nrmse.item())

        if epoch % 1000 == 0:
            print(f'{epoch=} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]}')
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title('MSE')
            plt.plot(losses_train, label='Train')
            plt.plot(losses_test, label='Test')
            plt.legend(loc='upper right')
            plt.xlabel('epoch x1000')
            plt.ylabel('loss')
            plt.subplot(2, 1, 2)
            plt.title('NRMSE')
            plt.plot(nrmse_train, label='Train')
            plt.plot(nrmse_test, label='Test')
            plt.legend(loc='upper right')
            plt.xlabel('epoch x1000')
            plt.ylabel('loss')
            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
