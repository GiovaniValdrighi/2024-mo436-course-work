import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class EncoderConv(nn.Module):
    def __init__(self, input_dim, encoding_dim, seq_len, h_dims, h_activ):
        super().__init__()
        self.conv_layers = []
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=h_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=h_dims[0],
            out_channels=h_dims[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc = nn.Linear(
            in_features=h_dims[1] * (seq_len // 2 // 2), out_features=encoding_dim
        )

        self.pool = nn.MaxPool1d(2, stride=2)
        self.activation = h_activ

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, encoding_dim, output_dim, seq_len, h_dims, h_activ, out_activ):
        super().__init__()
        self.h_dims = h_dims
        self.fc = nn.Linear(
            in_features=encoding_dim, out_features=h_dims[1] * (seq_len // 2 // 2)
        )

        self.deconv2 = nn.ConvTranspose1d(
            in_channels=h_dims[1],
            out_channels=h_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv1 = nn.ConvTranspose1d(
            in_channels=h_dims[0],
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.activation = h_activ
        self.out_activ = out_activ
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample2 = nn.Upsample(size=seq_len, mode="nearest")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.h_dims[1], -1)
        x = self.deconv2(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.deconv1(x)
        x = self.upsample2(x)
        x = self.out_activ(x)
        return x


class AutoencoderConv(nn.Module):
    def __init__(self, input_dim, encoding_dim, seq_len, h_dims, h_activ, out_activ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.seq_len = seq_len
        self.h_dims = h_dims
        self.h_activ = h_activ
        self.out_activ = out_activ

        self.encoder = EncoderConv(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            seq_len=seq_len,
            h_dims=h_dims,
            h_activ=h_activ,
        )

        self.decoder = DecoderConv(
            encoding_dim=encoding_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            h_dims=h_dims,
            h_activ=h_activ,
            out_activ=out_activ,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderMLP(nn.Module):
    def __init__(self, layers_dims):
        super().__init__()
        self.input_dim = layers_dims[0]
        self.encoding_dim = layers_dims[-1]
        encoder = []
        for i in range(len(layers_dims) - 1):
            encoder.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
            encoder.append(nn.ReLU())
        encoder = encoder[:-1]
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i in range(len(layers_dims) - 1, 0, -1):
            decoder.append(nn.Linear(layers_dims[i], layers_dims[i - 1]))
            decoder.append(nn.ReLU())
        decoder = decoder[:-1]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
