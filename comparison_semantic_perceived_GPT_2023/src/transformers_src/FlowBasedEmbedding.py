import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RealNVP Module
class RealNVP(nn.Module):
    def __init__(self, data_dim):
        super(RealNVP, self).__init__()
        
        # Create mask
        self.mask = torch.arange(data_dim).to(device) % 2

        # Create scale and translation networks
        self.scale_transform = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )

        self.translation_transform = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )
    
    def forward(self, x):
        scale = self.scale_transform(x * (1 - self.mask))
        translation = self.translation_transform(x * (1 - self.mask))

        y = self.mask * x + (1 - self.mask) * (x * torch.exp(scale) + translation)
        log_jacobian_det = scale.mean(dim=1).sum(dim=1)

        return y, log_jacobian_det


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.scale = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        activation = F.linear(x, self.weight, self.bias)
        return x + self.scale * torch.tanh(activation), -torch.log(1 - (self.scale * (1 - torch.tanh(activation) ** 2) * self.weight).pow(2) + 1e-6).sum(-1)


class RecurrentFlowEmbedding(nn.Module):
    def __init__(self, input_dim, num_layers, output_dim, device, dropout_rate=0.5):
        super(RecurrentFlowEmbedding, self).__init__()

        #LSTM to transform the input
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)  

        #PlanarFlow
        self.planar_flow = PlanarFlow(output_dim)

        # RealNVP
        self.realnvp = RealNVP(output_dim)  

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear layer to reduce dimensionality
        #self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.device = device
        
    def loss_function(self, z, log_jac_det):
        # Assuming a standard normal distribution as the target distribution
        log_likelihood = torch.distributions.Normal(0, 1).log_prob(z).mean(dim=1).sum(dim=1)
#         print(log_likelihood.size())
#         print(log_jac_det.size())
        # Using the change of variable formula for the loss
        loss = -(log_likelihood + log_jac_det).sum(dim=0)
        return loss

    def forward(self, x):
        # Pass x through the LSTM
        x, _ = self.lstm(x)
        x = self.dropout(x)  # Apply dropout after LSTM

        # Pass x through the PlanarFlow
        x, log_jac_det_planar = self.planar_flow(x)
        x = self.dropout(x)

        # Pass x through RealNVP
        z, log_jac_det_nvp = self.realnvp(x)

        # Sum the log determinants from Planar and RealNVP layers
        #log_jac_det = log_jac_det_planar + log_jac_det_nvp
        emb_loss = self.loss_function(z, log_jac_det_nvp)
        #print(emb_loss)
        # Return transformed data and log determinant of Jacobian
        return z.to(self.device), emb_loss.to(self.device)