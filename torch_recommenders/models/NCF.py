# Original codes are from
# https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/ncf.py

import torch
import torch.nn as nn

from .layer import *


class NeuralCollaborativeFiltering(nn.Module):
    """
    A pytorch implementation of Neural Collaborative Filtering.
    Reference:
        X He, et al. Neural Collaborative Filtering, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = nn.Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        x = self.embedding(x)
        user_x = x[:, 0] # x[:, [0]].squeeze(1)
        item_x = x[:, 1] # x[:, [1]].squeeze(1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)
        