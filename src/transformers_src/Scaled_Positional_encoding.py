import math
import torch
import torch.nn as nn


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(ScaledPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_length = seq_length

        self.register_buffer('position_weights', self._create_position_weights())
        self.position_scale1 = nn.Parameter(torch.ones(1))
        self.position_scale2 = nn.Parameter(torch.ones(1))

    def _create_position_weights(self):
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
        position_weights = torch.zeros(self.seq_length, self.d_model)
        position_weights[:, 0::2] = torch.sin(position * div_term)
        position_weights[:, 1::2] = torch.cos(position * div_term)
        return position_weights.unsqueeze(0)

    def forward(self, x, t):
        if t == 'e':
            pe = self.position_scale1 * self.position_weights[:, :x.size(1)]
        elif t == 'd':
            pe = self.position_scale2 * self.position_weights[:, :x.size(1)]
        return x+pe
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_length = seq_length

        self.register_buffer('position_weights', self._create_position_weights())

    def _create_position_weights(self):
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
        position_weights = torch.zeros(self.seq_length, self.d_model)
        position_weights[:, 0::2] = torch.sin(position * div_term)
        position_weights[:, 1::2] = torch.cos(position * div_term)
        return position_weights.unsqueeze(0)

    def forward(self, x):
        pe = self.position_weights[:, :x.size(1)]
        return x+pe