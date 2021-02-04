import cv2
import torch
import argparse
import numpy as np
from model import NN
from dataset import get_dataset
from utils import map_to_image, bgr_tensor_to_rgb_numpy


def visualize(filename, modelname, input_cols, target_cols, save_video=False):
    """Visualize the content of the generator along with the prediction made by the model."""
    dataset = get_dataset(filename, device='cpu', augment=False,
                          input_cols=input_cols, target_cols=target_cols)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fps', type=int,
                        help='frames per second for the video output', default=30)
    args = parser.parse_args()

    size = (400, 300)

    device = 'cpu'
    out_channels = 400
    model = NN(in_channels=3, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(modelname))
    model.eval()

    if save_video:
        video = cv2.VideoWriter(
            './out/prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), args.fps, (720, 660))
        print('Making the video...')

    print(dict(enumerate(dataset.cumulative_sizes)))

    for example in dataset.batches(1, shuffle=False):
        x, _, _, _, y = example
        y = y.reshape([20, 20])

        pred = model(x)
        pred = pred.reshape([20, 20]).detach().cpu()
        pred = pred.pow(0.6)

        frame = bgr_tensor_to_rgb_numpy(x.squeeze(0).detach())
        frame = frame[:, :, ::-1]  # rgb to bgr
        frame = cv2.resize(frame, size)
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        frame = (frame * 255).astype(np.uint8)

        gt_map = map_to_image(y, (15, 15), (3, 3))
        pred_map = map_to_image(pred, (15, 15), (3, 3))
        maps = np.hstack(
            [np.full((357, 2, 3), 255, dtype=np.uint8),
             gt_map,
             np.full((357, 2, 3), 255, dtype=np.uint8),
             pred_map,
             np.full((357, 2, 3), 255, dtype=np.uint8)])
        frame = np.hstack([
            np.full((300, 160, 3), 255, dtype=np.uint8),
            frame,
            np.full((300, 160, 3), 255, dtype=np.uint8)])
        frame = np.vstack(
            [maps, np.full((3, 720, 3), 255, dtype=np.uint8), frame])

        if save_video:
            video.write(frame)

        cv2.imshow('generator', frame)
        key = chr(cv2.waitKey(1000 // args.fps) & 0xFF)
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

    visualize(filename, modelname, input_cols, target_cols, save_video=True)
