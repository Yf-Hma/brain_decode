import torch.nn as nn
from .AddNorm import Norm
from .Multi_head_attention import MultiHeadAttention, MultiHeadDuplexAttention, MultiHeadSimplexAttention
from .ConvFeedForward import ConvFeedForward
from .FeedForward import FeedForward

class EncoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = ConvFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm_1(x)
        x = self.attn(x2, x2, x2, src_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class SimplexEncoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.s_attn = MultiHeadSimplexAttention(d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, y):
        x2 = self.norm_1(x)
        x = self.s_attn(x2, y)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, y

class DuplexEncoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.d_attn = MultiHeadDuplexAttention(d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, y):
        x2 = self.norm_1(x)
        x, y = self.d_attn(x2, y)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, y

class AlternatingEncoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.s_attn = MultiHeadSimplexAttention(d_model, heads)
        self.d_attn = MultiHeadDuplexAttention(d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, y):
        x2 = self.norm_1(x)
        x = self.s_attn(x2, y)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x, y = self.d_attn(x2, y)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, y

class AlternatingEncoderLayerConv2(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.s_attn = MultiHeadSimplexAttention(d_model, heads)
        self.d_attn = MultiHeadDuplexAttention(d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, y):
        x2 = self.norm_1(x)
        x, y = self.d_attn(x2, y)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = self.s_attn(x2, y)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm_1(x)
        x = self.attn(x2, x2, x2, src_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class AlternatingEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.2):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.s_attn = MultiHeadSimplexAttention(d_model, heads)
        self.d_attn = MultiHeadDuplexAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, y):
        x2 = self.norm_1(x)
        x = self.s_attn(x2, y)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x, y = self.d_attn(x2, y)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, y
