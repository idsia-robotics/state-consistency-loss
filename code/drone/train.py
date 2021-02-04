import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import PenguiNet
from tqdm import tqdm, trange
from datetime import datetime
from ConvBlock import ConvBlock
from torchsummary import summary
from sklearn.metrics import r2_score
from dataset import DroneDataset, transf_matrix


def transf_to_pose(t):
    return torch.tensor(
        [*t[:3, 3], torch.atan2(t[1, 0], t[0, 0])]).to(t.device)


def train():
    """Train the neural network model, save weights and plot loss over time."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_' + str(datetime.now()))
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs of the training phase', default=64)
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=64)
    parser.add_argument('-lr', '--learning-rate', type=float,
                        help='learning rate used for the training phase', default=7e-5)
    parser.add_argument('-t', '--delta-t', type=int,
                        help='delta-t used to define the window used for the overlap loss', default=0)
    parser.add_argument('-o', '--overlap', action='store_true',
                        help='if set enables the overlap loss function')
    parser.add_argument('-lmd', '--overlap-lambda', type=float,
                        help='lambda paramter used to weight the overlap loss', default=1e0)
    parser.add_argument('-sp', '--split-percentage', type=float,
                        help='t1 / t2 split percentage in [0, 1]', default=0.5)
    args = parser.parse_args()

    name = args.name
    epochs = args.epochs
    delta_t = args.delta_t
    half_batch_size = args.batch_size // 2
    learning_rate = args.learning_rate
    use_overlap = args.overlap
    overlap_lambda = args.overlap_lambda
    split_percentage = args.split_percentage * 0.8

    model_path = './drone/model/' + name
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

    train_dataset = DroneDataset('./drone', train=True, flip=True)

    # Model

    model = PenguiNet(ConvBlock, [1, 1, 1], True).to(device)
    summary(model, (1, 96, 160), device=device)

    # Optimizer & Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.L1Loss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1),
    #                                                        patience=3, verbose=False,
    #                                                        threshold=0.001, threshold_mode='rel', cooldown=1,
    #                                                        min_lr=0.1e-5, eps=1e-08)

    # Training

    split_index = int(split_percentage * len(train_dataset.frame_ids) - 1)
    val_index = int(0.8 * len(train_dataset.frame_ids) - 1)
    end_index = len(train_dataset.frame_ids) - 1

    steps_per_epoch = 300
    history = pd.DataFrame()
    old_mean_val_loss = float('inf')

    epochs_logger = trange(1, epochs + 1, desc='epoch')
    for epoch in epochs_logger:
        steps_logger = tqdm(enumerate(zip(
            train_dataset.timed_batches(half_batch_size,
                                        slice(0, split_index - delta_t),
                                        steps_per_epoch, t=delta_t),
            train_dataset.timed_batches(half_batch_size,
                                        slice(split_index, val_index - delta_t),
                                        steps_per_epoch, t=delta_t)
        )), desc='step', total=steps_per_epoch)
        val_logger = tqdm(train_dataset.batches(half_batch_size,
                                                slice(val_index, end_index),
                                                steps_per_epoch),
                          desc='val_step', total=steps_per_epoch)
        for step, ((x1, y1, _), (x2, _, t2)) in steps_logger:
            x1, y1, x2, t2 = x1.to(device), y1.to(
                device), x2.to(device), t2.to(device)
            model.train()
            optimizer.zero_grad()

            preds = model(x1)

            loss = loss_function(preds, y1)

            if use_overlap:
                preds = model(x2)
                pair_indices = torch.arange(preds.size(0)).view(-1, 2)
                pairs = preds[pair_indices]

                pair_first = pairs[:, 0, :]
                pair_second = pairs[:, 1, :]
                pair_second_transf = torch.stack(
                    [transf_matrix(t, lib=torch) for t in pair_second]).to(t2.device)

                pairs_transf = t2[pair_indices]
                rel_poses = torch.bmm(torch.inverse(pairs_transf[:, 0, ...]),
                                      pairs_transf[:, 1, ...])

                pair_second_adj_transf = torch.bmm(
                    rel_poses, pair_second_transf)

                pair_second_adj = torch.stack(
                    [transf_to_pose(t) for t in pair_second_adj_transf])

                overlap_loss = overlap_lambda * torch.nn.functional.mse_loss(
                    pair_first, pair_second_adj)
                loss += overlap_loss
                overlap = overlap_loss.item()
            else:
                overlap = None

            loss.backward()
            optimizer.step()

            loss = loss.item()

            if step == steps_per_epoch - 1:
                model.eval()
                with torch.no_grad():
                    val_r2 = 0
                    val_loss = 0
                    val_counter = 0
                    for val_x, val_y in val_logger:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_preds = model(val_x)

                        val_loss += loss_function(val_preds, val_y).item()

                        val_r2 += np.array([r2_score(val_y[:, i].cpu().numpy(),
                                                     val_preds[:, i].cpu().numpy()) for i in range(4)])
                        val_counter += 1

                    val_r2 /= val_counter
                    val_loss /= val_counter

                    # scheduler.step(val_loss)
            else:
                val_r2 = None
                val_loss = None

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'overlap': overlap,
                'val_r2': val_r2,
                'val_loss': val_loss
            }, ignore_index=True)

            mean_values = history.query('epoch == ' + str(epoch)).mean(axis=0)
            mean_loss = mean_values['loss']
            mean_val_r2 = mean_values['val_r2']
            mean_val_loss = mean_values['val_loss']

            log_str = 'loss: %.6f' % (mean_loss)
            steps_logger.set_postfix_str(log_str)
            if step == steps_per_epoch - 1:
                log_str += ', val_loss: %.6f, val_r2: %s' % (
                    mean_val_loss, str(mean_val_r2))
                print()
                epochs_logger.set_postfix_str(log_str)

        if old_mean_val_loss > mean_val_loss:
            checkpoint_name = '%d_%.6f_state_dict.pth' % (epoch, mean_val_loss)
            torch.save(model.state_dict(), checkpoint_path +
                       '/' + checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path + '/best.pth')
            old_mean_val_loss = mean_val_loss

    history.to_csv(output_path + '/history.csv')


if __name__ == "__main__":
    train()
