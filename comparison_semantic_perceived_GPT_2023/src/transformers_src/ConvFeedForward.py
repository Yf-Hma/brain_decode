import torch
import torch.nn as nn


class ConvFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(ConvFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=3, padding='same')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Apply the first convolutional layer
        x = self.conv1(x)

        # Apply the ReLU activation function
        x = self.relu(x)

        # Apply the second convolutional layer
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x
