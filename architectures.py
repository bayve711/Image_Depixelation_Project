import torch
import torch.nn as nn

class DualCNN(nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 9,  n_kernels: int = 128, kernel_size: int = 3):
        super(DualCNN, self).__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(nn.BatchNorm2d(n_kernels))
            cnn.append(nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = nn.Sequential(*cnn)

        self.output_layer = nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, pixel, known):
        outs_1 = torch.cat((pixel, known), dim=1)
        outs_2 = self.hidden_layers(outs_1)
        preds = self.output_layer(outs_2)

        preds = torch.clamp(preds, 0, 1)
        return preds

















