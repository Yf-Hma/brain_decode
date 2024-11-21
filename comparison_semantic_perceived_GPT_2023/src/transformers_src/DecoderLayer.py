import torch.nn as nn
from .AddNorm import Norm
from .Multi_head_attention import MultiHeadAttention, MultiHeadDuplexAttention, MultiHeadSimplexAttention
from .Multi_layer_Encoder_Decoder_Attention import MultiLayerAttention
from .ConvFeedForward import ConvFeedForward
from .FeedForward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, e_output, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x = self.attn(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn(x2, e_output, e_output, enc_dec_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class AlternatingDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadSimplexAttention(d_model, heads)
        self.attn_2 = MultiHeadDuplexAttention(d_model, heads)
        self.attn_3 = MultiLayerAttention(N, d_model, heads)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, y, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x = self.attn_1(x2, y, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x, y = self.attn_2(x2, y, trg_mask)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x2 + self.dropout_3(self.attn_3(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_4(x)
        x = x2 + self.dropout_4(self.ff(x2))
        return x, y


class DecoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiLayerAttention(N, d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)

    def forward(self, x, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class DuplexDecoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadDuplexAttention(d_model, heads)
        self.attn_2 = MultiLayerAttention(N, d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)

    def forward(self, x, y, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x, y = self.attn_1(x2, y, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x2 + self.dropout_2(self.attn_2(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_3(x)
        x = x2 + self.dropout_3(self.ff(x2))
        return x, y

class SimplexDecoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadSimplexAttention(d_model, heads)
        self.attn_2 = MultiLayerAttention(N, d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)

    def forward(self, x, y, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x = self.attn_1(x2, y, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x2 + self.dropout_2(self.attn_2(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_3(x)
        x = x2 + self.dropout_3(self.ff(x2))
        return x, y

class AlternatingDecoderLayerConv2(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadSimplexAttention(d_model, heads)
        self.attn_2 = MultiHeadDuplexAttention(d_model, heads)
        self.attn_3 = MultiLayerAttention(N, d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)

    def forward(self, x, y, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x, y= self.attn_2(x2, y, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x = self.attn_1(x2, y, trg_mask)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x2 + self.dropout_3(self.attn_3(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_4(x)
        x = x2 + self.dropout_4(self.ff(x2))
        return x, y

class AlternatingDecoderLayerConv(nn.Module):
    def __init__(self, d_model, d_ff, heads, N, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadSimplexAttention(d_model, heads)
        self.attn_2 = MultiHeadDuplexAttention(d_model, heads)
        self.attn_3 = MultiLayerAttention(N, d_model, heads)
        self.ff = ConvFeedForward(d_model, d_ff)

    def forward(self, x, y, e_outputs, enc_dec_mask, trg_mask):
        x2 = self.norm_1(x)
        x = self.attn_1(x2, y, trg_mask)
        x = x + self.dropout_1(x)
        x2 = self.norm_2(x)
        x, y = self.attn_2(x2, y, trg_mask)
        x = x + self.dropout_2(x)
        x2 = self.norm_3(x)
        x = x2 + self.dropout_3(self.attn_3(x2, e_outputs, enc_dec_mask))
        x2 = self.norm_4(x)
        x = x2 + self.dropout_4(self.ff(x2))
        return x, y
