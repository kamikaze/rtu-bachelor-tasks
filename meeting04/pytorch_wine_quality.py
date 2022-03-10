from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.datasets
import torch.nn
from torch.nn import Module, Sequential, Linear, Tanh, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

torch.set_default_dtype(torch.float64)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


class LossCrossEntropy(Module):
    def __init__(self):
        super().__init__()
        self.y = None
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim

        return torch.mean(-y * torch.log(y_prim))


def standardize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    stddev = np.std(values, axis=0)

    return (values - mean) / stddev


def get_f1_score(matrix: np.ndarray) -> dict:
    score = {}

    for item_idx in range(matrix.shape[0]):
        tp = matrix[item_idx, item_idx]
        tn = matrix[item_idx].sum() - tp
        fn = matrix[:, item_idx].sum() - tp
        fp = matrix.sum() - tp - tn - fn

        score[item_idx] = (2 * tp) / (2 * tp + fp + fn)

    return score


def get_confusion_matrix(expected, predicted) -> np.ndarray:
    matrix = np.zeros(shape=(3, 3), dtype=np.int)

    for expected_item, predicted_item in zip(expected, predicted):
        matrix[expected_item][predicted_item] += 1

    return matrix


class Model(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Linear(in_features=13, out_features=10, device=device),
            Tanh(),
            Linear(in_features=10, out_features=3, device=device),
            Softmax(dim=1)
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)

        return y_prim


class WineDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        return sample, label


def main():
    plt.show()

    features, classes = sklearn.datasets.load_wine(return_X_y=True)
    idx_rand = np.random.permutation(len(features))
    features = features[idx_rand]
    classes = classes[idx_rand]

    features = standardize(features)
    class_idxes = classes
    classes = np.zeros((len(classes), len(np.unique(classes))))
    classes[np.arange(len(classes)), class_idxes] = 1.0

    idx_split = int(len(features) * 0.85)
    dataset_train = WineDataset(
        torch.tensor(features[:idx_split], device=device),
        torch.tensor(classes[:idx_split], device=device)
    )
    dataset_test = WineDataset(
        torch.tensor(features[idx_split:], device=device),
        torch.tensor(classes[idx_split:], device=device)
    )

    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    epoch_count = 100_000
    learning_rate = 1e-6

    model = Model()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    losses_train = []
    losses_test = []
    accuracy_train = []
    accuracy_test = []
    tick_labels = ['0', '1', '2']

    loss_fn = LossCrossEntropy()

    for epoch in range(epoch_count+1):
        for dataset_loader in (train_dataloader, test_dataloader):
            if dataset_loader is train_dataloader:
                confusion_matrix_train = []
            else:
                confusion_matrix_test = []

            for x, y in dataset_loader:
                # Every FloatTensor has value/data and grad props as Variable class had
                y_prim = model.forward(x)
                loss = loss_fn.forward(y, y_prim)

                if dataset_loader is train_dataloader:
                    loss.backward()
                    optimizer.step()

                if epoch % 200 == 0:
                    # Let's stop calculating gradients
                    with torch.no_grad():
                        _, predicted = torch.max(y_prim.data, 1)
                        _, expected = torch.max(y.data, 1)
                        accuracy = (predicted == expected).sum() / y.size(0)

                        if dataset_loader is train_dataloader:
                            losses_train.append(loss.item())
                            accuracy_train.append(accuracy.item())
                            confusion_matrix_train.append(get_confusion_matrix(expected, predicted))
                        else:
                            losses_test.append(loss.item())
                            accuracy_test.append(accuracy.item())
                            confusion_matrix_test.append(get_confusion_matrix(expected, predicted))

        if epoch % 200 == 0:
            f1_score_train = get_f1_score(sum(confusion_matrix_train))
            f1_score_test = get_f1_score(sum(confusion_matrix_test))

            print(f'{epoch=} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]}')

            plt.clf()
            plt.subplot(2, 3, 1)
            plt.title('Cross-Entropy Loss')
            plt.plot(losses_train, label='Train')
            plt.plot(losses_test, label='Test')
            plt.legend(loc='upper right')
            plt.xlabel('epoch x200')
            plt.ylabel('loss')

            plt.subplot(2, 3, 2)
            plt.title('Train confusion matrix')
            ax = sns.heatmap(sum(confusion_matrix_train), annot=True, cmap='Blues')

            ax.set_title('Train confusion matrix')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Actual Values ')

            ax.xaxis.set_ticklabels(tick_labels)
            ax.yaxis.set_ticklabels(tick_labels)

            plt.subplot(2, 3, 3)
            plt.title('Train F1-score')
            plt.bar(f1_score_train.keys(), f1_score_train.values())

            plt.subplot(2, 3, 4)
            plt.title('Accuracy')
            plt.plot(accuracy_train, label='Train')
            plt.plot(accuracy_test, label='Test')
            plt.legend(loc='lower right')
            plt.xlabel('epoch x200')
            plt.ylabel('accuracy')

            plt.subplot(2, 3, 5)
            plt.title('Test confusion matrix')
            ax = sns.heatmap(sum(confusion_matrix_test), annot=True, cmap='Blues')

            ax.set_title('Test confusion matrix')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Actual Values ')

            ax.xaxis.set_ticklabels(tick_labels)
            ax.yaxis.set_ticklabels(tick_labels)

            plt.subplot(2, 3, 6)
            plt.title('Test F1-score')
            plt.bar(f1_score_test.keys(), f1_score_test.values())

            plt.draw()
            plt.pause(0.02)


if __name__ == '__main__':
    main()
