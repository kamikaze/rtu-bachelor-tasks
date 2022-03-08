import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torchvision
from torch.nn import ReLU, Linear, Parameter

torch.set_default_dtype(torch.float64)
matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()

has_cuda = torch.cuda.is_available()

DEVICE = torch.device('cuda:0' if has_cuda else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 1000
MAX_LEN = 0 if has_cuda else 200
INPUT_SIZE = 28


# TODO

class DatasetFassionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(root='../data', train=is_train, download=True)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN

        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x, dtype='float64')
        np_x = np.expand_dims(np_x, axis=0)
        x = torch.tensor(np_x, device=DEVICE)
        np_y = np.zeros((10,), dtype='float64')
        np_y[y_idx] = 1.0
        y = torch.tensor(np_y, device=DEVICE)

        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFassionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


def get_out_size(in_size: int, padding: int, kernel_size: int, stride: int) -> int:
    return int((in_size + 2*padding - kernel_size) / stride + 1)


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
                out_part = x_part @ K
                out[:, :, i_out, j_out] = out_part

                j_out += 1

            i_out += 1

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=1),
        )

        o_1 = get_out_size(INPUT_SIZE, kernel_size=5, stride=2, padding=1)
        o_2 = get_out_size(o_1, kernel_size=3, stride=2, padding=1)
        o_3 = get_out_size(o_2, kernel_size=3, stride=2, padding=1)

        self.fc = Linear(
            in_features=12*o_3*o_3,
            out_features=10
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
    model = Model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metrics = {}

    for stage in ['train', 'test']:
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

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
                    metrics_strs.append(f'{key}: {round(value, 2)}')

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
