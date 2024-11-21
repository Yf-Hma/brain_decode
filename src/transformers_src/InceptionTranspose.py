import torch
import torch.nn as nn

class InceptionTranspose(nn.Module):
    def __init__(self, in_channels, d_model):
        super(InceptionTranspose, self).__init__()

        self.branch1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, d_model, kernel_size=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        #self.out = nn.Linear(in_channels, d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        # Concatenate along time dimension
        output = torch.cat([branch1, branch2, branch3], 2)
        output = output.transpose(1,2) 
        return output