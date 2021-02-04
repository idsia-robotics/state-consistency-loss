import torch
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import DroneDataset


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
                        help='learning rate used for the training phase', default=5e-5)
    parser.add_argument('-o', '--overlap', action='store_true',
                        help='if set enables the overlap loss function')
    parser.add_argument('-lmd', '--overlap-lambda', type=float,
                        help='lambda paramter used to weight the overlap loss', default=1e0)
    parser.add_argument('-sp', '--split-percentage', type=float,
                        help='t1 / t2 split percentage in [0, 1]', default=0.4)
    args = parser.parse_args()

    name = args.name
    epochs = args.epochs
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

    # Dataset

    train_dataset = DroneDataset('./drone', train=True, flip=True)
    test_dataset = DroneDataset('./drone', train=False, flip=True)

    # Computation

    split_index = int(split_percentage * len(train_dataset.frame_ids) - 1)
    val_index = int(0.8 * len(train_dataset.frame_ids) - 1)
    end_index = len(train_dataset.frame_ids) - 1

    steps_per_epoch = 300

    for t in range(1, 10):
        plt.figure()
        yt = []

        steps_logger = tqdm(enumerate(zip(
            train_dataset.timed_batches(half_batch_size,
                                        slice(0, split_index - t),
                                        steps_per_epoch, t=t),
            train_dataset.timed_batches(half_batch_size,
                                        slice(split_index, val_index - t),
                                        steps_per_epoch, t=t)
        )), desc='step', total=steps_per_epoch)
        for _, ((x1, y1), (x2, y2)) in steps_logger:
            yt.append(y1.numpy())

        yt = np.concatenate(yt, axis=0)
        pair_indices = np.arange(yt.shape[0]).reshape(-1, 2)
        pairs = yt[pair_indices]

        sns.distplot((pairs[:, 0, :] - pairs[:, 1, :])
                     [:, 0], color='r', bins=20, label='x')
        sns.distplot((pairs[:, 0, :] - pairs[:, 1, :])
                     [:, 1], color='g', bins=20, label='y')
        sns.distplot((pairs[:, 0, :] - pairs[:, 1, :])[:, 3],
                     color='c', bins=20, label='$\phi$')
        plt.xlim(-5, 5)
        plt.ylim(0, 2.5)
        plt.legend()
        plt.title('Histogram of delta features of t0 - t' + str(t))
        plt.savefig(str(t) + '.png')

    plt.show()


if __name__ == "__main__":
    train()
