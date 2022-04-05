import argparse
import multiprocessing
import time
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.utils.data
from tqdm import tqdm

from meeting06.datasets.kaggle.face_masks import DatasetMasksNumpyMmap
from meeting06.models import InceptionNet


plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('dark_background')

# torch.set_default_dtype(torch.float64)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
WORKER_COUNT = 4#multiprocessing.cpu_count() if USE_CUDA else 0
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 8
LEARNING_RATE = 1e-8
EPOCHS = 100


class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        # return torch.mean(-y * torch.log(y_prim + 1e-8))
        return -torch.sum(y * torch.log(y_prim + 1e-20))


def run_epoch(data_loaders: Mapping, model, loss_func, optimizer, metric_keys, scaler=None):
    metrics_epoch = {key: [] for key in metric_keys}

    for stage, data_loader in data_loaders.items():
        for idx, (x, y) in enumerate(tqdm(data_loader)):
            x_on_device = x.to(device=DEVICE, dtype=torch.float16, non_blocking=True)
            y_on_device = y.to(device=DEVICE, dtype=torch.float16, non_blocking=True)

            with torch.cuda.amp.autocast():
                # Warning: consumes a lot of memory
                y_prim = model.forward(x_on_device)
                loss = loss_func.forward(y_on_device, y_prim)

            metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

            if stage == 'train':
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()

            # loss = loss.cpu()
            y_prim = y_prim.cpu()
            # x = x.cpu()
            # y = y.cpu()

            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)
            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

            if idx % 20 == 0:
                print(f"Loss {np.mean(metrics_epoch[f'{stage}_loss'])} "
                      f"Acc {np.mean(metrics_epoch[f'{stage}_acc'])}")

    return metrics_epoch


def benchmark_dataset(dataset, epochs):
    train_test_split = int(len(dataset) * TRAIN_TEST_SPLIT)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset,
        [train_test_split, len(dataset) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=USE_CUDA,
        num_workers=WORKER_COUNT
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=USE_CUDA,
        num_workers=WORKER_COUNT
    )

    model = InceptionNet()
    loss_func = LossCrossEntropy()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if USE_CUDA:
        model = model.to(DEVICE)
        # model = model.cuda()
        loss_func = loss_func.cuda()

    metrics = {}

    data_loaders = {'train': data_loader_train, 'test': data_loader_test}

    for stage in data_loaders.keys():
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, epochs):
        metrics_epoch = run_epoch(data_loaders, model, loss_func, optimizer, metrics.keys(), scaler)

        for key, values in metrics_epoch.items():
            mean_value = np.mean(values)
            metrics[key].append(mean_value)

        print(f'epoch: {epoch} {" ".join(f"{k} {v[-1]}" for k, v in metrics.items())}')

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


def get_parsed_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root', type=str)
    # parser.add_argument('--id', default=0, type=int)
    # parser.add_argument('--sequence_name', default='sequence', type=str)
    # parser.add_argument('--run_name', default='run', type=str)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--model', default='model_emo_VAE_v3', type=str)
    # parser.add_argument('--datasource', default='datasource_emo_v2', type=str)

    args, args_other = parser.parse_known_args()

    return args


def main():
    args = get_parsed_args()
    data_root_path = Path(args.data_root)
    datasets = [
        DatasetMasksNumpyMmap(root=data_root_path),
        # DatasetWikiArtFilesystem(root='../data'),
        # DatasetWikiArtNumpyMmap(root='../data'),
        # DatasetWikiArtCuPyMmap(root='../data'),
        # DatasetWikiArtZarr(root='../data', chunks=(BATCH_SIZE, None)),
    ]

    for dataset in datasets:
        class_name = dataset.__class__.__name__

        try:
            print(f'Loading {class_name}')
            dataset.load()
            start_time = time.perf_counter()
            print(f'{start_time}: Benchmarking {class_name}')
            benchmark_dataset(dataset, EPOCHS)
            end_time = time.perf_counter()
            print(f'{end_time}: Benchmark took {end_time - start_time}')
        except KeyboardInterrupt:
            print(f'Skipping {class_name}')


if __name__ == '__main__':
    main()
