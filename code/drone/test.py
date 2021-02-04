import torch
import numpy as np
from tqdm import tqdm
from model import PenguiNet
from ConvBlock import ConvBlock
from dataset import DroneDataset
from sklearn.metrics import r2_score


def test(name):
    device = 'cuda'
    modelname = f'./drone/model/{name}/checkpoints/best.pth'

    # Dataset

    test_dataset = DroneDataset('./drone', train=False)

    # Model

    model = PenguiNet(ConvBlock, [1, 1, 1], True).to(device)
    model.load_state_dict(torch.load(modelname))
    model.eval()
    loss_function = torch.nn.L1Loss()

    steps_per_epoch = 1000

    # Testing

    tes_logger = tqdm(test_dataset.batches(32,
                                           slice(
                                               0, len(test_dataset.frame_ids) - 1),
                                           steps_per_epoch),
                      desc='testing', total=steps_per_epoch)
    model.eval()
    with torch.no_grad():
        tes_r2 = 0
        tes_loss = 0
        tes_counter = 0
        for tes_x, tes_y in tes_logger:
            tes_x, tes_y = tes_x.to(device), tes_y.to(device)
            tes_preds = model(tes_x)

            tes_loss += loss_function(tes_preds, tes_y).item()

            tes_r2 += np.array([r2_score(tes_y[:, i].cpu().numpy(),
                                         tes_preds[:, i].cpu().numpy()) for i in range(4)])
            tes_counter += 1

        tes_r2 /= tes_counter
        tes_loss /= tes_counter

    print('TEST loss', tes_loss, ', r2', tes_r2)
    return name, tes_loss, tes_r2


if __name__ == "__main__":
    name = 'model_upperbound_r1'
    test(name)
