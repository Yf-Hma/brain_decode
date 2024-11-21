import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

