import torch
from torchsummary import summary


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NN, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=5,
                            padding=2, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5,
                            padding=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=5,
                            padding=2, stride=2),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(640, out_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = NN(3, 400)
    summary(model, (3, 64, 80), device='cpu')
