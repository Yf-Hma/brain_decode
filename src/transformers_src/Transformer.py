import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoders import Encoder, CNNEncoder, AlternatingEncoder, ConvAlternatingEncoder, FlowAlternatingEncoder, AlternatingEncoderConv, AlternatingEncoderConv2, LstmAlternatingEncoder, DuplexEncoderConv, SimplexEncoderConv, DeconvAlternatingEncoderConv
from .Decoders import Decoder, CNNDecoder, FasttextCNNDecoder, AlternatingDecoder, CNNAlternatingDecoder, ConvAlternatingDecoder, ConvAlternatingDecoder2, AlternatingDecoderFasttext, DuplexDecoderConv, SimplexDecoderConv

from .FlowBasedEmbedding import RecurrentFlowEmbedding

# Classic Transformer
class Transformer(nn.Module):
    def __init__(self, time_steps, src_fmri_features, max_seq_length, trg_vocab, d_model, d_ff, N, heads, device):
        super().__init__()
        self.encoder = Encoder(src_fmri_features, d_model, d_ff, N, heads, device)
        self.decoder = Decoder(trg_vocab, max_seq_length, d_model, d_ff, N, heads, device)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg):
        e_outputs, src_mask = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask)
        output = self.out(d_output)
        output_prob = F.softmax(output, dim=2)
        return output, output_prob

# CNN Transformer Hybrid
class CNNTransformer(nn.Module):
    def __init__(self, time_steps, src_fmri_features, max_seq_length, trg_vocab, d_model, d_ff, N, heads, device):
        super().__init__()
        self.encoder = CNNEncoder(src_fmri_features, d_model, d_ff, N, heads, device)
        self.decoder = CNNDecoder(trg_vocab, max_seq_length, d_model, d_ff, N, heads, device)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg):
        e_outputs, src_mask = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask)
        output = self.out(d_output)
        output_prob = F.softmax(output, dim=2)
        return output, output_prob


class DuplexTransformerConv(nn.Module):
    def __init__(self, time_steps, src_fmri_features, max_seq_length, trg_vocab, d_model, d_ff, N, heads, device):
        super().__init__()
        self.encoder = DuplexEncoderConv(time_steps, src_fmri_features, d_model, d_ff, N, heads, device)
        self.decoder = DuplexDecoderConv(trg_vocab, max_seq_length, d_model, d_ff, N, heads, device)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg):

        e_outputs, src_mask = self.encoder(src)

        d_output = self.decoder(trg, e_outputs, src_mask)
        output = self.out(d_output)
        output_prob = F.softmax(output, dim=2)
        return output, output_prob

# Bipartite Transformer with linear embedding and Con1D fully connected
class BipartiteTransformerConv(nn.Module):
    def __init__(self, time_steps, src_fmri_features, max_seq_length, trg_vocab, d_model, d_ff, N, heads, device):
        super().__init__()
        self.encoder = AlternatingEncoderConv(time_steps, src_fmri_features, d_model, d_ff, N, heads, device)
        self.decoder = ConvAlternatingDecoder(trg_vocab, max_seq_length, d_model, d_ff, N, heads, device)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg):
        e_outputs, src_mask = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask)
        output = self.out(d_output)
        output_prob = F.softmax(output, dim=2)
        return output, output_prob


#Bipartite transformer without an embedding layer:
class DeconvBipartiteTransformerConv(nn.Module):
    def __init__(self, time_steps, src_fmri_features, max_seq_length, trg_vocab, d_model, d_ff, N, heads, device):
        super().__init__()
        self.encoder = DeconvAlternatingEncoderConv(time_steps, src_fmri_features, d_model, d_ff, N, heads, device)
        self.decoder = ConvAlternatingDecoder(trg_vocab, max_seq_length, d_model, d_ff, N, heads, device)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg):
        e_outputs, src_mask = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask)
        output = self.out(d_output)
        output_prob = F.softmax(output, dim=2)
        return output, output_prob
