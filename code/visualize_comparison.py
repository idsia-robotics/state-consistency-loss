import cv2
import torch
import argparse
import numpy as np
from model import NN
from dataset import get_dataset
from utils import map_to_image, bgr_tensor_to_rgb_numpy


def visualize(filename, input_cols, target_cols, save_video=False):
    """Visualize the content of the generator along with the prediction made by the model."""
    dataset = get_dataset(filename, device='cpu', augment=False,
                          input_cols=input_cols, target_cols=target_cols)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fps', type=int,
                        help='frames per second for the video output', default=30)
    args = parser.parse_args()

    alpha = 0.4
    size = (400, 300)

    device = 'cpu'
    out_channels = 400

    bmodel = NN(in_channels=3, out_channels=out_channels).to(device)
    bmodel.load_state_dict(torch.load(
        './model/model_new_no/checkpoints/best.pth'))
    bmodel.eval()

    omodel = NN(in_channels=3, out_channels=out_channels).to(device)
    omodel.load_state_dict(torch.load(
        './model/model_new_o/checkpoints/best.pth'))
    omodel.eval()

    if save_video:
        video = cv2.VideoWriter(
            './out/comparison.mp4', cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (842, 842))
        print('Making the video...')

    starting_indices = dataset.cumulative_sizes
    print(dict(enumerate(starting_indices)))

    for batch in dataset.batches(128, shuffle=True):
        Xs, _, _, _, Ys = batch
        for x, y in zip(Xs, Ys):
            x = x.unsqueeze(0)
            y = y.reshape([20, 20])

            bpred = bmodel(x)
            bpred = bpred.reshape([20, 20]).detach().cpu()

            opred = omodel(x)
            opred = opred.reshape([20, 20]).detach().cpu()
            # opred = opred.pow(alpha)

            # normalize
            bm = bpred.mean()
            bv = bpred.std()
            om = opred.mean()
            ov = opred.std()

            opred = bv * ((opred - om) / ov + bm)
            opred = opred.clamp(0, 1)

            frame = bgr_tensor_to_rgb_numpy(x.squeeze(0).detach())
            frame = frame[:, :, ::-1]  # rgb to bgr
            frame = cv2.resize(frame, size)
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = (frame * 255).astype(np.uint8)

            gt_map = map_to_image(y, (17, 17), (4, 4))
            opred_map = map_to_image(opred, (17, 17), (4, 4))
            bpred_map = map_to_image(bpred, (17, 17), (4, 4))
            padded_frame = np.vstack([
                np.full((58, 400, 3), 255, dtype=np.uint8),
                frame,
                np.full((58, 400, 3), 255, dtype=np.uint8)])
            padded_frame = np.hstack([
                np.full((416, 8, 3), 255, dtype=np.uint8),
                padded_frame,
                np.full((416, 8, 3), 255, dtype=np.uint8)])
            maps = np.hstack([
                np.vstack([
                    padded_frame,
                    np.full((10, 416, 3), 255, dtype=np.uint8),
                    gt_map
                ]),
                np.full((416 * 2 + 10, 10, 3), 255, dtype=np.uint8),
                np.vstack([
                    bpred_map,
                    np.full((10, 416, 3), 255, dtype=np.uint8),
                    opred_map
                ])
            ])
            frame = maps

            if save_video:
                video.write(frame)

            cv2.imshow('generator', frame)
            key = chr(cv2.waitKey(1000 // args.fps) & 0xFF)
            if key == 'q':
                break
            elif key == 'n':
                break

        if key == 'q':
            break

    if save_video:
        video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = './dataset.h5'
    modelname = './model/model_new_o/checkpoints/best.pth'

    input_cols = ['camera', 'pos_x', 'pos_y', 'theta']
    target_cols = ['target_map']

    visualize(filename, input_cols, target_cols, save_video=True)
