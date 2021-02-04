import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def transf_matrix(pose, lib=np):
    t = lib.eye(4)
    t[:3, 3] = pose[:3]

    sin = lib.sin(pose[3])
    cos = lib.cos(pose[3])
    r = lib.eye(4)
    r[[0, 1], [0, 1]] = cos
    r[1, 0] = sin
    r[0, 1] = -sin

    if lib == torch:
        return lib.mm(t, r)

    return lib.dot(t, r)


class DroneDataset(Dataset):

    def __init__(self, root, train, flip=False):
        self.root = root
        self.flip = flip
        self.xcol = 'frame'
        self.ycol = 'rel_pose'
        self.filename = 'train' if train else 'test'
        self.filename += '.pickle'

        if not self.check_exists():
            raise RuntimeError('Dataset not found')

        self.df = pd.read_pickle(os.path.join(self.root,
                                              self.filename))
        self.frame_ids = np.unique(self.df['frame_id'])
        self.aug_ids = np.unique(self.df['aug_id'])

        self.df['drone_pose_matrix'] = self.df['drone_pose'].apply(
            transf_matrix)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.filename))

    def __getitem__(self, index):
        x = np.stack(self.df[self.xcol].values[index])
        y = np.stack(self.df[self.ycol].values[index])

        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        y = torch.tensor(y, dtype=torch.float, requires_grad=False)

        return x[:, None, :, :], y

    def backup_random_index_pair(self, frame_slice=..., aug_slice=...):
        frame_id = np.random.choice(self.frame_ids[frame_slice])
        aug_id = np.random.choice(self.aug_ids[aug_slice])
        idx = self.df[(self.df['frame_id'] == frame_id) &
                      (self.df['aug_id'] == aug_id)].index[0]
        return [idx, idx + 1]

    def random_index_pair(self, frame_slice=..., aug_slice=...):
        frame_id = np.random.choice(self.frame_ids[frame_slice])
        aug_id, new_aug_id = np.random.choice(self.aug_ids[aug_slice], size=2)
        idx = self.df[(self.df['frame_id'] == frame_id) &
                      (self.df['aug_id'] == aug_id)].index[0]
        new_idx = self.df[(self.df['frame_id'] == frame_id) &
                          (self.df['aug_id'] == new_aug_id)].index[0]
        return [idx, new_idx]

    def backup_batches(self, half_batch_size, frame_slice, how_many):
        for _ in range(0, how_many):
            idxs = [self.random_index_pair(frame_slice)
                    for _ in range(half_batch_size)]
            yield self[np.array(idxs).flatten()]

    def batches(self, half_batch_size, frame_slice, how_many):
        for _ in range(0, how_many):
            idxs = [self.random_index_pair(frame_slice)
                    for _ in range(half_batch_size)]
            x, y = self[np.array(idxs).flatten()]

            if self.flip and np.random.rand() > .5:
                x = torch.flip(x, (-1,))
                y[:, 1] = -y[:, 1]
                y[:, 3] = -y[:, 3]

            yield (x, y)

    def timed_batches(self, half_batch_size, frame_slice, how_many, t):
        for _ in range(0, how_many):
            idxs = [self.random_index_pair(frame_slice)
                    for _ in range(half_batch_size)]
            idxs = np.array(idxs)
            idxs[:, 1] += np.random.randint(0, t + 1)
            x, y = self[idxs.flatten()]

            drone_pose = np.stack(
                self.df['drone_pose_matrix'].values[idxs.flatten()])
            drone_pose = torch.tensor(
                drone_pose, dtype=torch.float, requires_grad=False)

            if self.flip and np.random.rand() > .5:
                x = torch.flip(x, (-1,))
                y[:, 1] = -y[:, 1]
                y[:, 3] = -y[:, 3]

            yield (x, y, drone_pose)
