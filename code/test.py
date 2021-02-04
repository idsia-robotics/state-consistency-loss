#!/usr/bin/python

"""Train the neural network model using the given training set."""

import os
import torch
import argparse
import numpy as np
from model import NN
import seaborn as sns
from settings import coords
import matplotlib.pyplot as plt
from dataset import get_dataset
from torchsummary import summary
from pytorchutils import MaskedAUROC


def test():
    """Train the neural network model, save weights and plot loss over time."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_new_o')
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the dataset (.h5 file)', default='./dataset.h5')
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=256)
    args = parser.parse_args()

    name = args.name
    filename = args.filename
    batch_size = args.batch_size

    out_channels = 400
    model_path = './model/' + name
    checkpoint_path = model_path + '/checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for k, v in vars(args).items():
        print('{0} = "{1}"'.format(k, v))
    print('device = "' + device + '"')

    if not os.path.exists(checkpoint_path):
        print('Model parameters not found: ' + checkpoint_path)
        exit()

    # Dataset

    input_cols = ['camera', 'pos_x', 'pos_y', 'theta']
    target_cols = ['target_map']
    train_test_split = 11

    dataset = get_dataset(filename, device=device, augment=False,
                          input_cols=input_cols, target_cols=target_cols)
    split_index = dataset.cumulative_sizes[train_test_split]

    # Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NN(in_channels=3, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(checkpoint_path + '/best.pth'))
    summary(model, (3, 64, 80), device=device)

    auc_function = MaskedAUROC()

    # Testing

    aucs = []
    for x, px, py, pt, y in dataset.batches(batch_size, start=split_index, shuffle=False):
        pose = torch.stack([px, py, pt], dim=-1).to(device)
        mask = y > -1

        preds = model(x)

        aucs.append(auc_function(preds, y, mask).cpu().numpy())

    auc = np.nanmean(aucs, axis=0).reshape(20, 20)
    auc = np.rot90(auc, 1)
    auc = np.fliplr(auc) * 100

    print('AUC: ' + str(auc.mean().item()))

    print(auc)

    rounded = (100 * coords).round(2).astype(int)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 5.8))
    sns.distplot(auc, bins=int(np.ceil(auc.max() - auc.min())),
                 ax=ax[0], kde=False, rug=False, color='red', hist_kws={'rwidth': 0.75})
    sns.heatmap(auc, cmap='gray', annot=True, cbar_kws={'shrink': .8},
                vmin=50, vmax=100, linewidths=0, ax=ax[1])
    plt.yticks(.5 + np.arange(20), np.unique(rounded[:, 0])[::-1])
    plt.xticks(.5 + np.arange(20), np.unique(rounded[:, 1]))
    plt.xlabel('Y [cm]')
    plt.ylabel('X [cm]')
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax[1].yaxis.get_majorticklabels(), rotation=0)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
