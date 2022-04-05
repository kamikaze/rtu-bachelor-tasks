import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from meeting06.datasets import DatasetFlickrImageNumpyMmap

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('dark_background')

torch.set_default_dtype(torch.float64)
USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
MAX_LEN = None if USE_CUDA else 200


class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)

        self.conv21 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)
        self.conv22 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(5, 5), stride=(stride, stride), padding=(2, 2), bias=False)

        self.conv31 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)
        self.conv32 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.conv33 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)

        self.avg_pool41 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv42 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels * 4)

    def forward(self, x):
        out1 = self.conv11.forward(x)

        out2 = self.conv21.forward(x)
        out2 = self.conv22.forward(out2)

        out3 = self.conv31.forward(x)
        out3 = self.conv32.forward(out3)
        out3 = self.conv33.forward(out3)

        out4 = self.avg_pool41(x)
        out4 = self.conv42(out4)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = F.relu(out)
        out = self.bn.forward(out)

        return out


class InceptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.inception_block1 = InceptionBlock(in_channels=32, out_channels=16)

        self.conv2 = torch.nn.Conv2d(in_channels=16 * 4, out_channels=128, kernel_size=(5, 5), stride=(1, 1),
                                     bias=False)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.inception_block2 = InceptionBlock(in_channels=128, out_channels=64)

        self.linear = torch.nn.Linear(64 * 4 * 61 * 61, 1)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.max_pool1.forward(out)
        out = F.relu(out)

        out = self.inception_block1.forward(out)

        out = self.conv2.forward(out)
        out = self.max_pool2.forward(out)
        out = F.relu(out)

        out = self.inception_block2.forward(out)

        # B, 256, 61, 61 -> 256x61x61
        out = out.view(-1, 256 * 61 * 61)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out


def main():
    dataset_full = DatasetFlickrImageNumpyMmap(root='../data')
    train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset_full,
        [train_test_split, len(dataset_full) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = InceptionNet()
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

                # Warning: consumes a lot of memory
                y_prim = model.forward(x)
                loss = loss_func.forward(y, y_prim)
                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # loss = loss.cpu()
                y_prim = y_prim.cpu()
                # x = x.cpu()
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
        plts = []
        c = 0

        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)
            plts += plt.plot(value, f'C{c}', label=key)
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        plt.tight_layout(pad=0.5)
        plt.draw()
        plt.pause(0.1)

    input('quit?')


if __name__ == '__main__':
    main()
