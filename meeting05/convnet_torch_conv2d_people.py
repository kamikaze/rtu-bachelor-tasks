import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
from sklearn.datasets import fetch_lfw_people
from torch.nn import ReLU, Linear, Parameter
from torch.utils.data import random_split

plt.rcParams['figure.figsize'] = (10, 5)
torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if has_cuda else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 1000
MAX_LEN = None if has_cuda else 200
INPUT_SIZE = 125


def standardize(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    stddev = np.std(values, axis=0)

    return (values - mean) / stddev


def get_out_size(in_size: int, padding: int, kernel_size: int, stride: int) -> int:
    return int((in_size + 2*padding - kernel_size) / stride + 1)


class DatasetLFWPeople(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        data = fetch_lfw_people(data_home='../data', color=False, slice_=None, min_faces_per_person=10)
        self.x = standardize(data.images)
        self.y = data.target
        self.class_count = max(self.y) + 1

    def __len__(self):
        return MAX_LEN if MAX_LEN else len(self.y)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x = self.x[idx]
        y_idx = self.y[idx]
        np_x = np.array(pil_x, dtype='float64')
        np_x = np.expand_dims(np_x, axis=0)
        x = torch.tensor(np_x, device=DEVICE)
        np_y = np.zeros((self.class_count,), dtype='float64')
        np_y[y_idx] = 1.0
        y = torch.tensor(np_y, device=DEVICE)

        return x, y


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = Parameter(
            torch.zeros(kernel_size, kernel_size, in_channels, out_channels, device=DEVICE)
        )
        torch.nn.init.kaiming_uniform_(self.K)

    def forward(self, x):
        batch_size = x.size(0)
        in_size = x.size(-1)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)
        out = torch.zeros(batch_size, self.out_channels, out_size, out_size, device=DEVICE)

        x_padded_size = in_size + self.padding * 2
        x_padded = torch.zeros(
            batch_size, self.in_channels, in_size+self.padding*2, x_padded_size, device=DEVICE
        )
        x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x

        # -1 means self.kernel_size*self.kernel_size*self.in_channels
        K = self.K.view(-1, self.out_channels)
        i_out = 0

        for i in range(0, x_padded_size - self.kernel_size, self.stride):
            j_out = 0

            for j in range(0, x_padded_size - self.kernel_size, self.stride):
                x_part = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # -1 means self.kernel_size*self.kernel_size*self.in_channels
                x_part = x_part.reshape(batch_size, -1)

                # size == (B, out_channels)
                # out_part = x_part @ K
                out_part = (K.t() @ x_part.unsqueeze(dim=-1)).squeeze(dim=-1)
                out[:, :, i_out, j_out] = out_part
                j_out += 1

            i_out += 1

        return out


class Model(torch.nn.Module):
    def __init__(self, class_count: int):
        super().__init__()

        out_channels = 16
        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )

        o_1 = get_out_size(INPUT_SIZE, kernel_size=3, stride=2, padding=1)
        o_2 = get_out_size(o_1, kernel_size=3, stride=2, padding=1)
        o_3 = get_out_size(o_2, kernel_size=3, stride=2, padding=1)

        self.fc = Linear(
            in_features=out_channels*o_3*o_3,
            out_features=class_count
        )

    def forward(self, x):
        # returns B, C_in, W_in, H_in
        batch_size = x.size(0)
        # returns B, C_out, W_out, H_out
        out = self.encoder.forward(x)
        # returns B, F
        out_flat = out.view(batch_size, -1)
        # returns B, 10
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)

        return y_prim


def main():
    dataset = DatasetLFWPeople()
    dataset_size = len(dataset)
    train_len = int(0.8 * dataset_size)
    test_len = dataset_size - train_len
    dataset_train, dataset_test = random_split(dataset, (train_len, test_len))

    model = Model(dataset.class_count)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metrics = {}

    for stage in ['train', 'test']:
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for epoch in range(1, EPOCHS):
        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            stage = 'train'

            if data_loader == data_loader_test:
                stage = 'test'

            for x, y in data_loader:
                y_prim = model.forward(x)
                loss = torch.mean(-y * torch.log(y_prim + 1e-8))

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                np_y_prim = y_prim.cpu().data.numpy()
                np_y = y.cpu().data.numpy()
                idx_y = np.argmax(np_y, axis=1)
                idx_y_prim = np.argmax(np_y_prim, axis=1)
                acc = np.average((idx_y == idx_y_prim) * 1.0)
                metrics_epoch[f'{stage}_acc'].append(acc)
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            metrics_strs = []

            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {value}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plts = []
        c = 0

        plt.clf()

        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.draw()
        plt.pause(0.05)


if __name__ == '__main__':
    main()
