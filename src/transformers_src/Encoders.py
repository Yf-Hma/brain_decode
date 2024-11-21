import torch.nn as nn
import torch
import copy
from .CNNEmbedding import CNNEmbedding
from .LstmConvEmbedding import LstmConvEmbedding
from .Scaled_Positional_encoding import ScaledPositionalEncoding, PositionalEncoding
from .EncoderLayer import EncoderLayer, EncoderLayerConv, AlternatingEncoderLayer, AlternatingEncoderLayerConv, AlternatingEncoderLayerConv2, DuplexEncoderLayerConv, SimplexEncoderLayerConv
from .AddNorm import Norm
from .FlowBasedEmbedding import RecurrentFlowEmbedding
from .InceptionTranspose import InceptionTranspose


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Normal transormer encoder with linear layer instead of embedding
class Encoder(nn.Module):
    def __init__(self, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = PositionalEncoding(d_model, feature_number)
        self.layers = get_clones(EncoderLayer(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2).to(self.device)

    def forward(self, src):
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x)
        H = []
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

# CNN transformer encoder with Conv1D embedding and Conv1D feedforward
class CNNEncoder(nn.Module):
    def __init__(self, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = CNNEmbedding(feature_number, d_model, kernel_size=3, stride=1)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(EncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2).to(self.device)

    def forward(self, src):
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
            x = self.norm(x)
            H.append(x)
        return H, src_mask


# Bipartite encoder with Linear Embedding + simplex then duplex attention and linear feed forward
class AlternatingEncoder(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayer(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

# Bipartite encoder with Lstm and Conv Embedding + simplex then duplex attention and linear feed forward
class LstmAlternatingEncoder(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = LstmConvEmbedding(feature_number, d_model, kernel_size=3, stride=1, hidden_size=256, num_layers=1)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayer(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask


# Bipartite encoder with Conv1D Embedding + simplex then duplex attention + Conv1D feed forward
class ConvAlternatingEncoder(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = CNNEmbedding(feature_number, d_model, kernel_size=3, stride=1)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

class SimplexEncoderConv(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(SimplexEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

class DuplexEncoderConv(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(DuplexEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

# Bipartite encoder with linear embedding + simplex then duplex attention + Conv1D feed forward
class AlternatingEncoderConv(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

class AlternatingEncoderConv2(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayerConv2(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

#Bipartite Transformer with Deconv Inception model as an embedding + con1d feedforward:
class DeconvAlternatingEncoderConv(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = InceptionTranspose(feature_number, d_model)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)
    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        x = self.embed(src)
        src_mask = self.create_src_mask(x)
        x = self.pe(x, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask

# Bipartite Transformer with simplex then duplex and Flow based embedding and Conv1D feed forward
class FlowAlternatingEncoder(nn.Module):
    def __init__(self, time_steps, feature_number, d_model, d_ff, hidden_dim, num_layers, N, heads, device):
        super().__init__()
        self.N = N
        self.embed = RecurrentFlowEmbedding(feature_number, hidden_dim, num_layers, d_model, device, dropout_rate=0.5)
        self.pe = ScaledPositionalEncoding(d_model, feature_number)
        self.layers = get_clones(AlternatingEncoderLayerConv(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(time_steps, d_model))
        self.device = device

    def create_src_mask(self, src):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = torch.all(src != padding_token_id, dim=-1)  # Create a mask where masked tokens are False
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(self, src):
        y = self.y.expand(src.size(0), -1, -1)
        # Pass src through the RecurrentFlow
        x_emb = self.embed(src)
        src_mask = self.create_src_mask(x_emb)
        x = self.pe(x_emb, 'e')
        H = []
        for i in range(self.N):
            x, y = self.layers[i](x, y)
            x = self.norm(x)
            H.append(x)
        return H, src_mask
