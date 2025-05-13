import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

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


class MultiLayerAttention(nn.Module):
    def __init__(self, num_layers, d_model, heads):
        super(MultiLayerAttention, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # Define learnable weights for the attention mechanism
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.Wv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.Wi = nn.ModuleList([nn.Linear(d_model*2, 1) for _ in range(num_layers)])
        self.bi = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Z, encoding_outputs, mask=None):
        # Z: Output of the previous decoder module
        # encoding_outputs: List of encoding layer outputs [H1, H2, ..., HN]
        # mask: Optional mask tensor (shape: [batch_size, seq_length])
        bs = Z.size(0)
        attention_outputs = []
        for i in range(self.num_layers):
            Hi = encoding_outputs[i]
            #print(f'Hi size: {Hi.size()}')
            Q = self.Wq(Z).view(bs, -1, self.h, self.d_k)
            K = self.Wk[i](Hi).view(bs, -1, self.h, self.d_k)
            V = self.Wv[i](Hi).view(bs, -1, self.h, self.d_k)

            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            alpha_i = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)/  math.sqrt(self.d_k)), dim=-1)
            if mask is not None:
                #masks = mask.unsqueeze(1).unsqueeze(2)
                alpha_i = alpha_i.masked_fill(mask == 0, -1e9)

            attention_i = torch.matmul(alpha_i, V).view(bs, -1, self.d_model)
            attention_outputs.append(attention_i)
        weighted_multi_atts = []
        for i in range (self.num_layers):
            alpha = self.sigmoid(self.Wi[i](torch.cat((Z,attention_outputs[i]), dim=2))+ self.bi[i])
            weighted_multi_att = alpha * attention_outputs[i]
            weighted_multi_atts.append(weighted_multi_att)
        # Combine attention outputs from all layers
        weighted_multi_atts_sum = torch.sum(torch.stack(weighted_multi_atts), dim=0)
        attention_multi_layer_output = self.out(weighted_multi_atts_sum)
        return attention_multi_layer_output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, masks, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if masks is not None:
            scores = scores.masked_fill(masks == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask):

        bs = q.size(0)
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, mask, self.d_k, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class MultiHeadDuplexAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadDuplexAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, -1)

    def apply_mask(self, attn_weights, mask):
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        return attn_weights

    def forward(self, X, Y, mask=None):
        Q = self.q(X)
        K, V = self.k(Y), self.v(Y)

        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)

        # Compute attention from X to Y
        attn_weights = Q @ K.transpose(-2, -1) / self.d_k**0.5
        attn_weights = self.apply_mask(attn_weights, mask)
        attn_weights = F.softmax(attn_weights, dim=-1)

        Y = attn_weights @ V
        Y = self.combine_heads(Y)

        Y = self.gamma(Y) * self.d_k**0.5 + self.beta(Y)
        Y = self.out(Y)

        Q = self.q(Y)
        K, V = self.k(X), self.v(X)

        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)

        # Compute attention from Y to X
        attn_weights = Q @ K.transpose(-2, -1) / self.d_k**0.5
        attn_weights = self.apply_mask(attn_weights, mask)
        attn_weights = F.softmax(attn_weights, dim=-1)

        X = attn_weights @ V
        X = self.combine_heads(X)

        X = self.gamma(X) * self.d_k**0.5 + self.beta(X)
        X = self.out(X)

        return X, Y

class MultiHeadDuplexAttention1(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadDuplexAttention1, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Multi-head attention parameters
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.out = nn.Linear(d_model, d_model)

        # Linear layers for the update rule
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_len = x.size(1)
        return x.view(batch_size,seq_len , self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, Q, K, V, mask=None):
        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = attn_weights @ V
        return output, attn_weights

    def omega(self, X):
        mu = torch.mean(X, dim=1, keepdim=True)
        sigma = torch.std(X, dim=1, keepdim=True)
        return (X - mu) / (sigma + 1e-5)

    def compute_K(self, X, V):
        # Assuming the attention mechanism is used to compute K
        Q = self.q(X)
        K_V = self.k(V)
        V_V = self.v(V)

        _, attn_weights = self.attention(Q, K_V, V_V)
        return attn_weights @ X

    def ud(self, X, Y, mask):
        batch_size = X.size(0)

        Q = self.split_heads(self.q(X), batch_size)
        K = self.split_heads(self.compute_K(Y, X), batch_size)
        V = self.split_heads(Y, batch_size)

        A, _ = self.attention(Q, K, V, mask)
        A = A.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        A = self.out(A)

        return self.gamma(A) * self.omega(X) + self.beta(A)

    def ua(self, X, Y, mask):
        batch_size = X.size(0)

        Q = self.split_heads(self.q(X), batch_size)
        K, V = self.split_heads(self.k(Y), batch_size), self.split_heads(self.v(Y), batch_size)

        A, _ = self.attention(Q, K, V, mask)
        A = A.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        A = self.out(A)

        return F.layer_norm(X + A, X.shape[-1:])

    def forward(self, X, Y, mask=None):
        # Update Y based on X
        Y_new = self.ua(Y, X, mask)

        # Update X based on Y
        X_new = self.ud(X, Y_new, mask)

        return X_new, Y_new


class MultiHeadSimplexAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSimplexAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, -1)

    def apply_mask(self, attn_weights, mask):
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        return attn_weights

    def forward(self, X, Y, mask=None):

        Q = self.q(X)
        #print(Q.size())
        K, V = self.k(Y), self.v(Y)
        #print(K.size())

        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)

        attn_weights = Q @ K.transpose(-2, -1) / self.d_k**0.5
        #print(f'A {attn_weights.size()}')
        attn_weights = self.apply_mask(attn_weights, mask)
        #print(f'A {attn_weights.size()}')
        attn_weights = F.softmax(attn_weights, dim=-1)
        Y = attn_weights @ V
        Y = self.combine_heads(Y)

        # Apply scale and bias controlled by attended information
        X = self.gamma(Y) * (X - X.mean(dim=-1, keepdim=True)) / (X.std(dim=-1, keepdim=True) + 1e-9) + self.beta(Y)

        return self.out(X)

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
            nn.ConvTranspose1d(in_channels, d_model, kernel_size=5, padding=1),
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
