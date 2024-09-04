import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_sizes=((1, 5), (1, 3)),
        stride=(1, 1),
        dilation=(1, 1),
        padding="same",
        pool=(1, 10),
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size=kernel_sizes[0],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(pool)
        self.net = nn.Sequential(self.conv1, self.relu, self.pooling, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        return self.relu(out)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_sizes=((1, 5), (1, 3)),
        strides=((1, 1), (1, 1)),
        dilations=((1, 1), (1, 15)),
        padding="same",
        pooling=((1, 10), (1, 10)),
        dense_layers=(400, 200, 20),
        dropout=0.5,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(kernel_sizes)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_sizes[i],
                    stride=strides[i],
                    dilation=dilations[i],
                    padding=padding,
                    pool=pooling[i],
                )
            ]

        self.convolution = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # Dynamically determine the number of features after flattening
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, num_inputs, 250
            )  # (batch_size, n_channels, window_length)
            dummy_output = self.convolution(dummy_input)
            n_flatten = dummy_output.view(1, -1).size(1)
        dense_layers = (n_flatten,) + dense_layers

        layers = []
        for i in range(len(dense_layers) - 1):
            layers.append(
                nn.Linear(in_features=dense_layers[i], out_features=dense_layers[i + 1])
            )
            layers.append(nn.ReLU)
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features=dense_layers[-1], out_features=1))
        self.fc = nn.Sequential(*layers)
        self.network = nn.Sequential(self.convolution, self.flatten, self.fc)

    def forward(self, x):
        return self.network(x)

    def reset_model_weights(self):
        for layer in self.network.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
