import torch.nn.functional as F
from params import args
from torch import nn
from utils import *
import torch as t
import math

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform

class Classifier(nn.Module):
    """
    Input: embedding of EEG samples
    Method: simple Multi Layer Perceptron
    Output: predictions (batch_size, 26)
    """
    def __init__(self, hidden_dim=128, num_classes=26):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mlp1 = nn.Linear(self.hidden_dim, 64)
        self.mlp2 = nn.Linear(self.hidden_dim, 64)
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, emb1, emb2):
        # emb1 = self.mlp1(emb1)
        # emb2 = self.mlp2(emb2)
        final_emb = t.cat([emb1, emb2], dim=-1)
        pred = self.mlp3(final_emb)
        return pred

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_):
        shortcut = self.shortcut(input_)
        input_ = nn.ReLU()(self.bn1(self.conv1(input_)))
        input_ = nn.ReLU()(self.bn2(self.conv2(input_)))
        input_ = input_ + shortcut
        return nn.ReLU()(input_)

class ResNetEncoder(nn.Module):
    def __init__(self, inpChannel=1, resblock=ResBlock):
        super(ResNetEncoder, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(inpChannel, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(32, 64, downsample=True),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = t.nn.AdaptiveAvgPool2d(1)

    def forward(self, input_):
        input_ = self.layer0(input_)
        input_ = self.layer1(input_)
        # input_ = self.layer2(input_)
        # input_ = self.layer3(input_)
        # input_ = self.layer4(input_)
        input_ = self.gap(input_)
        input_ = t.squeeze(input_)

        return input_

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, feature_dim=24):
        super(TemporalTransformerEncoder, self).__init__()
        # self.pos_emb = nn.Embedding(args.len_time, 24)
        self.pos_emb = get_pos_emb(args.len_time, 24).cuda()
        self.layers = nn.Sequential(
            TransformerLayer(in_dim=feature_dim, out_dim=32,  num_heads=4),
            TransformerLayer(in_dim=32,          out_dim=64,  num_heads=4),
            TransformerLayer(in_dim=64,          out_dim=128, num_heads=4)
        )

    def forward(self, x):
        embeds = [x]
        # embeds = [x + self.pos_emb.weight]
        # embeds = [x + self.pos_emb]
        for layer in self.layers:
            embeds.append(layer(embeds[-1]))
        return t.mean(embeds[-1], axis=1)

class SpatialTransformerEncoder(nn.Module):
    def __init__(self, feature_dim=args.len_time):
        super(SpatialTransformerEncoder, self).__init__()
        self.layers = nn.Sequential(
            TransformerLayer(in_dim=feature_dim, out_dim=256,  num_heads=3),
            TransformerLayer(in_dim=256,         out_dim=128,  num_heads=4),
            TransformerLayer(in_dim=128,         out_dim=128,  num_heads=4)
        )

    def forward(self, x):
        embeds = [x]
        for layer in self.layers:
            embeds.append(layer(embeds[-1]))
        return t.mean(embeds[-1], axis=1)

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=24):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.Sequential(
            TransformerLayer(in_dim=feature_dim, out_dim=32,  num_heads=4),
            TransformerLayer(in_dim=32,          out_dim=64, num_heads=4)
        )

    def forward(self, x):
        embeds = [x]
        for layer in self.layers:
            embeds.append(layer(embeds[-1]))
        return t.mean(embeds[-1], axis=1)

class TransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttentionLayer(in_dim, num_heads, dropout_prob=0.3)
        self.intermediate = IntermediateLayer(in_dim, out_dim, dropout_prob=0.1)

    def forward(self, x):
        attention_output = self.attention(x)
        intermediate_output = self.intermediate(x)
        return intermediate_output

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout_prob=0.3):
        super(SelfAttentionLayer, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = t.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class IntermediateLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.1):
        super(IntermediateLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.GELU(),
            # nn.Linear(in_dim, out_dim, bias=True),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.layers(x)

