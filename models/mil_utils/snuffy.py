# MIT License
#
# Copyright (c) 2020 Bin Li
# Copyright (c) 2024 Hossein Jafarinia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import math

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
from torch import nn
#  snuffy.py
import copy
import argparse




class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BClassifier(nn.Module):
    def __init__(self, encoder, num_classes, input_size: int):
        super(BClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        x, attentions = self.encoder(x, c)
        return self.linear(x.mean(dim=1)), attentions


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attntions = layer(x, c)
        return self.norm(x), attntions


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, c, top_big_lambda_indices, random_indices, mode):
        "Apply residual connection to any sublayer with the same size."
        if mode == 'attn':
            top_big_lambdas = torch.index_select(x, dim=1, index=top_big_lambda_indices)
            random_big_lambda = torch.index_select(x, dim=1, index=random_indices) if random_indices != None else None
            top_big_lambda = torch.cat((top_big_lambdas, random_big_lambda),
                                       dim=1) if random_indices != None else top_big_lambdas
            multiheadedattn = sublayer(self.norm(x))
            return top_big_lambda + self.dropout(multiheadedattn[0]), multiheadedattn[1]
        elif mode == 'ff':
            return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout, big_lambda, random_patch_share):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.big_lambda = big_lambda
        self.random_patch_share = random_patch_share
        self.top_big_lambda_share = 1.0 - random_patch_share

    def forward(self, x, c):
        "Follow Figure 1 (left) for connections."
        _, m_indices = torch.sort(c, 1, descending=True)
        top_big_lambda_share_indices = m_indices[:, 0:math.ceil(self.big_lambda * self.top_big_lambda_share),
                                       :].squeeze()
        top_big_lambdas = torch.index_select(x, dim=1, index=top_big_lambda_share_indices)

        if top_big_lambda_share_indices.dim() == 0:  # If topk_share_indices is a Scalar tensor, convert it to 1-D tensor
            top_big_lambda_share_indices = top_big_lambda_share_indices.unsqueeze(0)

        remaining_indices = list(set(range(x.shape[1])) - set(top_big_lambda_share_indices.tolist()))
        randoms_share = min(
            int(self.big_lambda * self.random_patch_share),
            max(0, x.shape[1] - math.ceil(self.big_lambda * self.top_big_lambda_share))
        )
        random_indices = torch.from_numpy(
            np.random.choice(remaining_indices, randoms_share, replace=False)).to(
            device) if randoms_share != 0 else None

        random_big_lambda = torch.index_select(x, dim=1, index=random_indices) if randoms_share != 0 else None
        top_big_lambda = torch.cat((top_big_lambdas, random_big_lambda),
                                   dim=1) if randoms_share != 0 else top_big_lambdas
        x_big_lambda, attentions = self.sublayer[0](x, lambda x: self.self_attn(x, top_big_lambda, x), c,
                                                    top_big_lambda_share_indices,
                                                    random_indices, 'attn')

        selected_indices = torch.hstack(
            (top_big_lambda_share_indices, random_indices)) if randoms_share != 0 else top_big_lambda_share_indices
        y = x.clone()

        #  TODO: whats happening here? pls check again. 
        y[:, selected_indices, :] = x_big_lambda

        return self.sublayer[1](y, self.feed_forward, c, None, None, 'ff'), attentions


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_big_lambda = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_big_lambda)
    p_attn = scores.softmax(dim=-1)

    value = value / torch.linalg.norm(value, dim=-1, keepdims=True)

    if dropout is not None:
        p_attn = dropout(p_attn)
        
    return torch.matmul(p_attn.transpose(-2, -1), value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_big_lambda = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_big_lambda).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(  # be in topk bedam
            query, key, value, dropout=self.dropout
        )
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_big_lambda)
        )
        del query
        del key
        del value

        return self.linears[-1](x), self.attn


class PositionwiseFeedForward(nn.Module):  # mikham
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, activation, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        activation_dictionary = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'selu': nn.SELU()
        }
        self.activation = activation_dictionary[activation]

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A = self.b_classifier(feats, classes)

        return classes, prediction_bag, A



class MILNet_batch(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super().__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        batch_feats, batch_classes = self.i_classifier(x)

        prediction_bags = []
        As = []

        for feats, classes in zip(batch_feats, batch_classes):
            prediction_bag, A = self.b_classifier(feats.unsqueeze(0), classes.unsqueeze(0))

            prediction_bags.append(prediction_bag)
            As.append(A)

        prediction_bags = torch.cat(prediction_bags, dim=0)
        As = torch.cat(As, dim=0)
        
        return batch_classes, prediction_bags, As
    


class SnuffyClass(nn.Module):

    def __init__(self, feats_size=384, num_classes=1, depth=1, mlp_multiplier=4,
                 activation='relu', big_lambda=900, encoder_dropout=0.1, l2normed_embeddings=1,
                 num_heads=4, random_patch_share=0.7777777777777778, 
                 weight_init__weight_init_i__weight_init_b=['trunc_normal', 'xavier_uniform', 'trunc_normal'],
                 soft_average=0
                 ):

        super().__init__()

        self.feats_size = feats_size
        self.num_classes = num_classes
        self.depth = depth
        self.mlp_multiplier=mlp_multiplier
        self.activation = activation
        self.big_lambda = big_lambda
        self.encoder_dropout=encoder_dropout
        self.l2normed_embeddings=l2normed_embeddings
        self.num_heads = num_heads
        self.random_patch_share=random_patch_share
        self.weight_init__weight_init_i__weight_init_b=weight_init__weight_init_i__weight_init_b
        self.soft_average=soft_average

        i_classifier = FCLayer(in_size=self.feats_size,
                                        out_size=self.num_classes).to(device)

        c = copy.deepcopy
        attn = MultiHeadedAttention(
            self.num_heads,
            self.feats_size,
        ).to(device)
        ff = PositionwiseFeedForward(
            self.feats_size,
            self.feats_size * self.mlp_multiplier,
            self.activation,
            self.encoder_dropout
        ).to(device)
        b_classifier = BClassifier(
            Encoder(
                EncoderLayer(
                    self.feats_size,
                    c(attn),
                    c(ff),
                    self.encoder_dropout,
                    self.big_lambda,
                    self.random_patch_share
                ), self.depth
            ),
            self.num_classes,
            self.feats_size
        ).to(device)

        self.milnet = MILNet_batch(i_classifier, b_classifier).to(device)
        self.sort_init_function()

        self.single_weight_parameter = self._get_single_weight_parameter()

    def _get_single_weight_parameter(self):
        rr = True if self.soft_average==1 else False
        single_weight_parameter = torch.tensor(0.5, requires_grad=rr, device=device)
        print('single_weight_parameter.requires_grad:', single_weight_parameter.requires_grad)
        single_weight_parameter.data.clamp_(0, 1)
        return single_weight_parameter


    def sort_init_function(self):

        init_funcs_registry = {
            'trunc_normal': nn.init.trunc_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'orthogonal': nn.init.orthogonal_
        }

        modules = [(self.weight_init__weight_init_i__weight_init_b[1], 'i_classifier'),
                    (self.weight_init__weight_init_i__weight_init_b[2], 'b_classifier')]
        print('modules:', modules)
        for init_func_name, module_name in modules:
            init_func = init_funcs_registry.get(init_func_name)
            print('init_func:', init_func)
            for name, p in self.milnet.named_parameters():
                if p.dim() > 1 and name.split(".")[0] == module_name:
                    init_func(p)

  
    def forward(self, x):

        if len(x.shape)==4:
            x = x.squeeze(-2)

        if self.l2normed_embeddings:
            x = x / torch.linalg.norm(x, axis=-1, keepdims=True)

        ins_prediction, bag_prediction, _ = self.milnet(x)
        max_prediction, _ = torch.max(ins_prediction, 1)

        return bag_prediction.squeeze(-1), max_prediction.squeeze(-1)

        # # TODO : 
        # """
        # fix from here: 26th Oct 2024
        # """
        # bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        # max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        # loss = self.single_weight_parameter * bag_loss + (1 - self.single_weight_parameter) * max_loss

        # with torch.no_grad():
        #     bag_prediction = (
        #             (1 - self.single_weight_parameter) * torch.sigmoid(max_prediction) +
        #             self.single_weight_parameter * torch.sigmoid(bag_prediction)
        #     ).squeeze().cpu().numpy()

        # return bag_prediction, loss, ins_prediction

    


if __name__ == '__main__':

    model = SnuffyClass().cuda()
    x = torch.randn((8, 1024, 384)).cuda()

    y = model(x)

    print(y)
