import torch.nn as nn
import torch
import copy
from .InceptionTranspose import InceptionTranspose
from .CNNEmbedding import CNNEmbedding
from .Scaled_Positional_encoding import ScaledPositionalEncoding, PositionalEncoding
from .DecoderLayer import DecoderLayer, DecoderLayerConv, AlternatingDecoderLayer, AlternatingDecoderLayerConv, AlternatingDecoderLayerConv2, DuplexDecoderLayerConv, SimplexDecoderLayerConv
from .AddNorm import Norm
from .FlowBasedEmbedding import RecurrentFlowEmbedding


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Classic Transformer Decoder with linear feed forward and normal encoder decoder attention
class Decoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(DecoderLayer(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        x = self.decoder_embedding(trg.long())
        x = self.pe(x)
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs[-1], src_mask, trg_mask)
        return self.norm(x)


# CNN Transformer Decoder with multi layer encoder decoder attention and Conv1D feed forward
class CNNDecoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(DecoderLayerConv(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

# CNN Transformer Decoder with multi layer encoder decoder attention, Conv1D feed forward and pre trained fasttext for embedding
class FasttextCNNDecoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, embedding_matrix, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.decoder_embedding.load_state_dict({'weight': embedding_matrix})
        self.decoder_embedding.weight.requires_grad = False
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(DecoderLayerConv(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        #print(trg_mask.size())
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

#Bipartite Decoder with simplex then duplex attention + multi layer attention and linear feed forward
class AlternatingDecoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(AlternatingDecoderLayer(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class SimplexDecoderConv(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(SimplexDecoderLayerConv(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class DuplexDecoderConv(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length - 1)
        self.layers = get_clones(DuplexDecoderLayerConv(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length - 1, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)

        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)

#Bipartite Decoder with simplex then duplex attention + multi layer attention and Conv1D feed forward
class ConvAlternatingDecoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length - 1)
        self.layers = get_clones(AlternatingDecoderLayerConv(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length - 1, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class ConvAlternatingDecoder2(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(AlternatingDecoderLayerConv2(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)

#Bipartite Decoder with simplex then duplex attention + multi layer attention and linear feed forward
class CNNAlternatingDecoder(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, device):
        super().__init__()
        self.N = N
        self.decoder_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(AlternatingDecoderLayer(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.y = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.decoder_embedding(trg.long())
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)


#Bipartite Decoder with simplex then duplex attention + multi layer attention and linear feed forward + fasttext pre trained embedding
class AlternatingDecoderFasttext(nn.Module):
    def __init__(self, vocab_len, max_seq_length, d_model, d_ff, N, heads, embedding_matrix, device):
        super().__init__()
        self.N = N
        self.fasttext_embedding = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.fasttext_embedding.load_state_dict({'weight': embedding_matrix})
        self.fasttext_embedding.weight.requires_grad = False
        self.y = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.pe = ScaledPositionalEncoding(d_model, max_seq_length)
        self.layers = get_clones(AlternatingDecoderLayer(d_model, d_ff, heads, N), N)
        self.norm = Norm(d_model)
        self.device = device

    def create_padding_mask(self, trg):
        padding_token_id = 0  # Assuming 0 represents the padding token ID
        mask = (trg != padding_token_id)  # Create a mask where pad tokens are True
        return mask.to(self.device)

    def create_lookahead_mask(self, trg_seq):
        trg_len = trg_seq.size(1)
        mask = 1 - torch.triu(torch.ones(trg_len, trg_len), diagonal=1)
        return mask.to(self.device)

    def create_combined_mask(self, trg):
        # Create padding mask
        padding_mask = self.create_padding_mask(trg)
        lookahead_mask = self.create_lookahead_mask(trg)
        lookahead_mask = lookahead_mask.unsqueeze(0)
        lookahead_mask = lookahead_mask.repeat(trg.size(0), 1, 1)

        padding_start = []
        for mask in padding_mask:
            mask = mask.float()
            occ = []
            for i in range(mask.size(0)):
                if mask[i].item() == 0:
                    occ.append(i)
            if len(occ) != 0:
                padding_start.append(occ[0])
            else:
                padding_start.append(0)
        for j in range(lookahead_mask.size(0) - 1):
            r = padding_start[j]
            if r != 0:
                lookahead_mask[j:, r:, :] = 0 * lookahead_mask[j:, r:, :]
        return lookahead_mask.unsqueeze(1).to(self.device)

    def forward(self, trg, e_outputs, src_mask):
        x = self.fasttext_embedding(trg.long())
        y = self.y.expand(trg.size(0), -1, -1)
        x = self.pe(x, 'd')
        trg_mask = self.create_combined_mask(trg)
        for i in range(self.N):
            x, y = self.layers[i](x, y, e_outputs, src_mask, trg_mask)
        return self.norm(x)
