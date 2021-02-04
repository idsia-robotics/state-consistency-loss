#!/usr/bin/python

"""Train the neural network model using the given training set."""

import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import NN
from settings import coords
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import get_dataset
from torchsummary import summary
from pytorchutils import MaskedLoss, MaskedAUROC


def train():
    """Train the neural network model, save weights and plot loss over time."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_' + str(datetime.now()))
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the dataset (.h5 file)', default='./dataset.h5')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs of the training phase', default=50)
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=64)
    parser.add_argument('-lr', '--learning-rate', type=float,
                        help='learning rate used for the training phase', default=5e-5)
    parser.add_argument('-o', '--overlap', action='store_true',
                        help='if set enables the overlap loss function')
    parser.add_argument('-lmd', '--overlap-lambda', type=float,
                        help='lambda paramter used to weight the overlap loss', default=1e0)
    args = parser.parse_args()

    name = args.name
    filename = args.filename
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    use_overlap = args.overlap
    overlap_lambda = args.overlap_lambda

    out_channels = len(coords)
    model_path = './model/' + name
    output_path = model_path + '/output'
    checkpoint_path = model_path + '/checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for k, v in vars(args).items():
        print('{0} = "{1}"'.format(k, v))
    print('device = "' + device + '"')

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

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
    summary(model, (3, 64, 80), device=device)

    # Optimizer & Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    auc_function = MaskedAUROC()
    loss_function = MaskedLoss(torch.nn.MSELoss)
    overlap_loss_function = OverlapLoss(
        coords=torch.tensor(coords, device=device, dtype=torch.float),
        distance=0.01)

    # plot utils

    def plot_lines(df, columns, colors, ax, alpha=0.25, show_range=False, window_size=1):
        for color, column in zip(colors, columns):
            agg_df = df.groupby('epoch')[column]

            if window_size > 1:
                agg_df = agg_df.rolling(window_size)

            means = agg_df.mean()
            ax.plot(np.arange(len(means)), means, c=color)

            if show_range:
                mins = agg_df.min()
                maxs = agg_df.max()
                ax.fill_between(x=np.arange(len(means)),
                                y1=mins, y2=maxs, alpha=alpha)

        ax.legend(columns)
        plt.ylim(0, 0.04)

    # Training

    history = pd.DataFrame()
    epochs_logger = trange(1, epochs + 1, desc='epoch')
    for epoch in epochs_logger:
        steps_logger = tqdm(dataset.batches(batch_size, stop=split_index),
                            desc='step', total=split_index // batch_size)
        for step, batch in enumerate(steps_logger):
            model.train()
            x, px, py, pt, y = batch
            pose = torch.stack([px, py, pt], dim=-1).to(device)
            mask = (y > -1).to(y.dtype)

            optimizer.zero_grad()

            preds = model(x)

            loss = loss_function(preds, y, mask)

            if use_overlap:
                overlap_loss = overlap_lambda * overlap_loss_function(
                    preds, pose)
                loss += overlap_loss
                overlap = overlap_loss.item()
            else:
                overlap = None

            loss.backward()
            optimizer.step()

            loss = loss.item()

            if step == (split_index // batch_size - 1):
                model.eval()
                with torch.no_grad():
                    val_auc = 0
                    val_loss = 0
                    val_counter = 0
                    for val_batch in dataset.batches(256, start=split_index):
                        val_x, _, _, _, val_y = val_batch
                        val_mask = val_y > -1

                        val_preds = model(val_x)

                        val_loss += loss_function(
                            val_preds, val_y, val_mask.to(val_y.dtype)).item()

                        varrr = auc_function(val_preds, val_y, val_mask)
                        val_auc += varrr[~torch.isnan(varrr)].mean().item()

                        val_counter += 1

                    val_auc /= val_counter
                    val_loss /= val_counter
            else:
                val_auc = None
                val_loss = None

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'overlap': overlap,
                'val_auc': val_auc,
                'val_loss': val_loss
            }, ignore_index=True)

            mean_values = history.query('epoch == ' + str(epoch)).mean(axis=0)
            mean_loss = mean_values['loss']
            mean_val_auc = mean_values['val_auc']
            mean_val_loss = mean_values['val_loss']

            log_str = 'loss: %.6f' % (mean_loss)
            steps_logger.set_postfix_str(log_str)
            if step == split_index // batch_size - 1:
                log_str += ', val_loss: %.6f, val_auc: %.6f' % (
                    mean_val_loss, mean_val_auc)
                print()
                epochs_logger.set_postfix_str(log_str)

        checkpoint_name = '%d_%.6f_state_dict.pth' % (epoch, mean_val_loss)
        torch.save(model.state_dict(), checkpoint_path + '/' + checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path + '/best.pth')

    history.to_csv(output_path + '/history.csv')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_lines(history, ['loss', 'val_loss'], ['blue', 'orange'], ax)
    plt.savefig(output_path + '/loss.svg')
    plt.show()


if __name__ == '__main__':
    train()
