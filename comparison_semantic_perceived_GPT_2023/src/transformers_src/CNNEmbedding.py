import torch.nn as nn


class CNNEmbedding(nn.Module):
    def __init__(self, feature_number, smaller_feature_number, kernel_size, stride):
        super(CNNEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(feature_number, smaller_feature_number, kernel_size, stride, padding='same')
        self.conv2 = nn.Conv1d(smaller_feature_number, smaller_feature_number, kernel_size, stride, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # Permute input tensor to (batch_size, feature_number, time_steps)
        x = x.permute(0, 2, 1)

        # Apply first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)

        # Apply second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)

        # Permute output tensor back to (batch_size, time_steps, smaller_feature_number)
        x = x.permute(0, 2, 1)

        return x
