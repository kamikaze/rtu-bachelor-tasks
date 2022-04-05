import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.hub import download_url_to_file
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('dark_background')

# torch.set_default_dtype(torch.float64)
USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_LEN = None if USE_CUDA else 200


class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            x, y, self.labels = pickle.load(fp)

        x = torch.from_numpy(np.array(x))
        self.x = x.permute(0, 3, 1, 2)
        self.input_size = self.x.size(-1)
        y = torch.LongTensor(y)
        self.y = F.one_hot(y)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx] / 255
        y = self.y[idx]

        return x, y


class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=(stride, stride), bias=False)
        else:
            self.is_bottleneck = False

    def forward(self, x):
        residual = x
        out = self.conv1.forward(x)
        out = F.relu(out)
        out = self.bn1.forward(out)
        out = self.conv2.forward(out)

        if self.is_bottleneck:
            residual = self.shortcut.forward(x)

        out += residual

        out = F.relu(out)
        out = self.bn2.forward(out)

        return out


class DenseBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.bn3 = torch.nn.BatchNorm2d(num_features=96)
        self.bn4 = torch.nn.BatchNorm2d(num_features=128)

        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), bias=False)

    def forward(self, x):
        out = F.relu(x)
        out = self.bn1.forward(out)
        conv1 = self.conv1.forward(out)

        conv2_in = torch.cat([x, conv1], dim=1)
        out = F.relu(conv2_in)
        out = self.bn2.forward(out)
        conv2 = self.conv2.forward(out)

        conv3_in = torch.cat([x, conv1, conv2], dim=1)
        out = F.relu(conv3_in)
        out = self.bn3.forward(out)
        conv3 = self.conv3.forward(out)

        conv4_in = torch.cat([x, conv1, conv2, conv3], dim=1)
        out = F.relu(conv4_in)
        out = self.bn4.forward(out)
        conv4 = self.conv4.forward(out)

        out = torch.cat([x, conv1, conv2, conv3, conv4], dim=1)

        return out


class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1), bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv.forward(x)
        out = F.relu(out)
        out = self.bn.forward(out)
        out = self.avg_pool.forward(out)

        return out


class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                    bias=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_block1 = DenseBlock()
        self.dense_block2 = DenseBlock()
        self.dense_block3 = DenseBlock()

        self.transition_layer1 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_layer2 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_layer3 = TransitionLayer(in_channels=160, out_channels=32)

        self.linear = torch.nn.Linear(288, 5)

    def forward(self, x):
        out = self.conv.forward(x)
        out = self.max_pool.forward(out)

        out = self.dense_block1.forward(out)
        out = self.transition_layer1.forward(out)

        out = self.dense_block2.forward(out)
        out = self.transition_layer2.forward(out)

        out = self.dense_block3.forward(out)
        out = self.transition_layer3.forward(out)

        # B, 32, 3, 3 -> 32x3x3 -> 288
        out = out.view(-1, 32 * 3 * 3)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # P = (F - 1) / 2 = (7 - 1) / 2 = 3
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=4)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # make ResBlocks
        self.identity_block_1 = ResBlock(in_channels=4, out_channels=4)
        self.identity_block_2 = ResBlock(in_channels=4, out_channels=4)

        self.bottleneck_block1 = ResBlock(in_channels=4, out_channels=8, stride=2)
        self.identity_block_3 = ResBlock(in_channels=8, out_channels=8)

        self.bottleneck_block2 = ResBlock(in_channels=8, out_channels=16, stride=2)
        self.identity_block_4 = ResBlock(in_channels=16, out_channels=16)

        self.bottleneck_block3 = ResBlock(in_channels=16, out_channels=32, stride=2)
        self.identity_block_5 = ResBlock(in_channels=32, out_channels=32)

        self.linear = torch.nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = F.relu(out)
        out = self.bn1.forward(out)
        out = self.max_pool.forward(out)

        out = self.identity_block_1.forward(out)
        out = self.identity_block_2.forward(out)

        out = self.bottleneck_block1.forward(out)
        out = self.identity_block_3.forward(out)

        out = self.bottleneck_block2.forward(out)
        out = self.identity_block_4.forward(out)

        out = self.bottleneck_block3.forward(out)
        out = self.identity_block_5.forward(out)

        # (64, N, 4, 4) -> (64, N, 1, 1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=5)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        out = out.view(x.size(0), -1)
        out = F.softmax(out, dim=1)

        return out


def main():
    # x = torch.randn(size=(64, 3, 100, 100))
    # model = ResNet()
    # y = model.forward(x)
    # print(y.shape)
    # exit()

    dataset_full = DatasetApples()
    train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset_full,
        [train_test_split, len(dataset_full) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    model = DenseNet()
    loss_func = LossCrossEntropy()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    if USE_CUDA:
        model = model.to('cuda:0')
        # model = model.cuda()
        loss_func = loss_func.cuda()

    metrics = {}

    for stage in ['train', 'test']:
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, 100):
        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            stage = 'train'

            if data_loader == data_loader_test:
                stage = 'test'

            for x, y in tqdm(data_loader):
                # #CAUTION random resize here!!! model must work regardless
                # out_size = int(28 * (random.random() * 0.3 + 1.0))
                # x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(out_size, out_size))

                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()

                y_prim = model.forward(x)
                loss = loss_func.forward(y, y_prim)
                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # loss = loss.cpu()
                y_prim = y_prim.cpu()
                x = x.cpu()
                y = y.cpu()

                np_y_prim = y_prim.data.numpy()
                np_y = y.data.numpy()

                idx_y = np.argmax(np_y, axis=1)
                idx_y_prim = np.argmax(np_y_prim, axis=1)

                acc = np.average((idx_y == idx_y_prim) * 1.0)
                metrics_epoch[f'{stage}_acc'].append(acc)

            metrics_strs = []

            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {value}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plt.clf()
        plt.subplot(121)  # row col idx
        plts = []
        c = 0

        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
            plt.subplot(3, 6, j)
            color = 'green' if idx_y_prim[i] == idx_y[i] else 'red'
            plt.title(f"pred: {idx_y_prim[i]}\n real: {idx_y[i]}", c=color)
            plt.imshow(x[i].permute(1, 2, 0))

        plt.tight_layout(pad=0.5)
        plt.draw()
        plt.pause(0.1)

    input('quit?')


if __name__ == '__main__':
    main()
