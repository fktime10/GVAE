import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .gcn_deconv import GCNDeconv
import torch.nn as nn
 

class GALA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, encoder_layers):
        super(GALA, self).__init__()

        # Encoder: Use GCNConv layers
        self.encoder_layers = torch.nn.ModuleList()
        self.encoder_layers.append(GCNConv(in_channels, encoder_layers[0]))
        for i in range(1, len(encoder_layers)):
            self.encoder_layers.append(GCNConv(encoder_layers[i - 1], encoder_layers[i]))
        self.encoder_layers.append(GCNConv(encoder_layers[-1], hidden_channels))

        # Decoder: Use GCNDeconv layers (from the separate gcn_deconv file)
        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_layers.append(GCNDeconv(hidden_channels, encoder_layers[-1]))
        for i in range(len(encoder_layers) - 1, 0, -1):
            self.decoder_layers.append(GCNDeconv(encoder_layers[i], encoder_layers[i - 1]))
        self.decoder_layers.append(GCNDeconv(encoder_layers[0], in_channels))

    def encode(self, x, edge_index):
        for layer in self.encoder_layers:
            x = F.relu(layer(x, edge_index))
        return x

    def decode(self, z, edge_index):
        for layer in self.decoder_layers[:-1]:
            z = F.relu(layer(z, edge_index))
        z = self.decoder_layers[-1](z, edge_index)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z, edge_index)
        return x_hat, z
