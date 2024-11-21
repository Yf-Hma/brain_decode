import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        """
        d_model: the number of features (usually 512 or 768 in popular Transformer models)
        dff: dimension of the feed-forward network, typically larger than d_model (e.g., 2048)
        """
        super(FeedForward, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(d_model, dff)

        # Second fully connected layer
        self.fc2 = nn.Linear(dff, d_model)

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_length, d_model)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out