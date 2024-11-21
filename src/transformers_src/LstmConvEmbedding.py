import torch.nn as nn

class LstmConvEmbedding(nn.Module):
    def __init__(self, feature_number, smaller_feature_number, kernel_size, stride, hidden_size, num_layers=1):
        super(LstmConvEmbedding, self).__init__()

        # Initial LSTM layer
        self.lstm1 = nn.LSTM(feature_number, hidden_size, num_layers, batch_first=True)

        # Convolutional layers
        self.conv1 = nn.Conv1d(hidden_size, smaller_feature_number, kernel_size, stride, padding='same')
        self.conv2 = nn.Conv1d(smaller_feature_number, smaller_feature_number, kernel_size, stride, padding='same')
        self.relu = nn.ReLU()

        # LSTM layer after convolutional layers
        self.lstm2 = nn.LSTM(smaller_feature_number, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Passing input through the initial LSTM layer
        x, (hn, cn) = self.lstm1(x)

        # Permute input tensor to (batch_size, feature_number, time_steps)
        x = x.permute(0, 2, 1)

        # Apply first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)

        # Apply second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)

        # Permute output tensor to (batch_size, time_steps, smaller_feature_number)
        x = x.permute(0, 2, 1)

        # Passing data through the second LSTM layer
        x, (hn, cn) = self.lstm2(x)

        return x

# Here, `hidden_size1` and `hidden_size2` are the number of features in the hidden state h of each of the LSTM layers. You can set these according to your needs.
