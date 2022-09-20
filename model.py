import torch.nn.functional as F
from params import args
from torch import nn
import torch as t

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            # todo
        )

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x