import math
import torch
import torch.nn as nn


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