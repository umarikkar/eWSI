# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for attention mechanisms."""

from math import ceil
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn.modules import Module

from einops import rearrange, reduce
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf

from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

import matplotlib.pyplot as plt

class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )



def _moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6):
    """Compute the Moore-Penrose pseudo-inverse of a tensor [1]_.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    iters: int = 6
        Number of iterations for Moore-Penrose algorithm.

    References
    ----------
    .. [1] G. Strang. "Linear Algebra and Its Applications, 2nd Ed."
           Academic Press, Inc., 1980, pp. 139-142.
    """
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    id_arr = torch.eye(x.shape[-1], device=device)
    id_arr = rearrange(id_arr, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = (
            0.25
            * z
            @ (13 * id_arr - (xz @ (15 * id_arr - (xz @ (7 * id_arr - xz)))))
        )

    return z


class NystromAttention(Module):
    """Nyström approximation for the Multi-Head Self-Attention.

    This code is derived from the nystrom-attention library:
    ``nystrom-attention``: https://github.com/mlpen/Nystromformer/tree/main (MIT License)

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 1.

    qkv_bias : bool = False
        Whether to add a bias to the linear projection for query, key and value.

    num_landmarks : int = 256
        Dimension of the landmarks used to approximate the matrix multiplication
        query-key (QK^T) in the Nyström method. When `nys_num_landmarks` is small,
        the approximation of the self-attention with the Nyström method scales
        linearly with the length of the input sequence.

    pinv_iterations : int = 6
        Number of iterations for the iterative Moore-Penrose pseudoinverse
        approximation.

    residual : bool = True
        Whether to implement a skip connexion for values V (with a depthwise
        convolution). See also the `residual_kernel_size` parameter. Defaults
        to True.

    residual_kernel_size : int = 33
        Kernel size for the 2D depthwise convolution used in the skip
        connexion of value V (to help convergence of the Nyström approximation).

    attn_dropout : Optional[float] = None
        Unused. For compatibility with the `SelfAttention` module.

    proj_dropout : float = 0
        Dropout rate (applied after the multiplication with the values).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_kernel_size: int = 33,
        attn_dropout: Optional[float] = None,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.__build()

    def __build(self):
        """Build the `NystromAttention` module."""
        head_dim = self.in_features // self.num_heads
        self.scale = head_dim**-0.5
        self.to_qkv = nn.Linear(
            self.in_features, self.in_features * 3, bias=self.qkv_bias
        )
        self.to_out = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.Dropout(self.proj_dropout),
        )
        if self.residual:
            _padding = (self.residual_kernel_size // 2, 0)
            self.res_conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(self.residual_kernel_size, 1),
                padding=_padding,
                groups=self.num_heads,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, seq_len, in_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, seq_len, in_features).
        """
        _, n, _, h, m, iters = (
            *x.shape,
            self.num_heads,
            self.num_landmarks,
            self.pinv_iterations,
        )

        # Pad so that sequence can be evenly divided into m landmarks
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

        # Derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        q = q * self.scale

        # Generate landmarks by sum reduction, and then calculate mean using the mask
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=ceil(n / m))
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=ceil(n / m))
        q_landmarks /= ceil(n / m)
        k_landmarks /= ceil(n / m)

        # Similarities
        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # Eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(
            lambda t: t.softmax(dim=-1), (sim1, sim2, sim3)
        )
        attn2_inv = _moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # Add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # Merge and combine heads
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        return out


class SelfAttention(Module):
    """Multi-Head Self-Attention.

    Implementation adapted from https://github.com/rwightman/pytorch-image-models.

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 1.

    qkv_bias : bool = False
        Whether to add a bias to the linear projection for query, key and value.

    attn_dropout : float = 0.0
        Dropout rate (applied before the multiplication with the values).

    proj_dropout : float = 0.0
        Dropout rate (applied after the multiplication with the values).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.__build()

    def __build(self):
        """Build the `SelfAttention` module."""
        head_dim = self.in_features // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(
            self.in_features, self.in_features * 3, bias=self.qkv_bias
        )
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.proj = nn.Linear(self.in_features, self.in_features)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, seq_len, in_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, seq_len, in_features).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        metadata=None, 
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v, mask=mask)
        
        attention_logits /= self.temperature

        return attention_logits

    def forward(
        self, v: torch.Tensor, metadata=None,  mask: Optional[torch.BoolTensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, metadata=metadata, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)

        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights





class JustAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(JustAttention, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        attention_logits = self.w(h_v, mask=mask) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
    



class VariAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(VariAttention, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.p = nn.Parameter(torch.Tensor([1, 1, 0]), requires_grad=True)
        self.q = nn.Parameter(torch.Tensor([1, 1, 0]), requires_grad=True)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        p = self.p
        q = self.q

        h_v = self.att(v)
        h_v = p[0]*torch.tanh(p[1]*h_v) + p[2]

        u_v = self.gate(v)
        u_v = q[0]*torch.tanh(q[1]*u_v) + q[2]

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
    



class VariAttention2(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(VariAttention2, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.p = nn.Parameter(torch.Tensor([1, 1, 0]), requires_grad=True)
        self.q = nn.Parameter(torch.Tensor([1, 1, 0]), requires_grad=True)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        p = self.p
        q = self.q

        h_v = self.att(v)
        h_v = p[0]*torch.tanh(p[1]*h_v) + p[2]

        u_v = self.gate(v)
        u_v = q[0]*torch.sigmoid(q[1]*u_v) + q[2]

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature
        return attention_logits

    def forward(    
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
    


class MaxPooler(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.m = nn.Identity()

    def forward(    
        self, v, dim=1
    ):

        v = self.m(v)
        
        dim = dim if type(dim)==list else [dim]

        for d in dim:
            v = v.max(dim=d, keepdim=True)[0]

        return v.squeeze(1)
    
    
class MeanPooler(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.m = nn.Identity()

    def forward(    
        self, v, dim=1
    ):

        v = self.m(v)
        
        dim = dim if type(dim)==list else [dim]

        for d in dim:
            v = v.mean(dim=d, keepdim=True)

        return v.squeeze(1)
    

class SeNet2D(torch.nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.fc_spatial_sq = nn.Linear(1024, 256)
        self.fc_spatial_ex = nn.Linear(256, 1024)

        self.fc_channel_sq = torch.nn.Conv2d(d_model, d_model // 4, 1, 1)
        self.fc_channel_ex = torch.nn.Conv2d(d_model // 4, d_model, 1, 1)

        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        metadata=None, 
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:


        v_interp = interpolate_coords(v, metadata)

        v_in = rearrange(v_interp, 'b h w d -> b d h w')

        b, d, h, w = v_in.shape

        v10 = torch.mean(v_in, dim=1, keepdim=True)

        sz=32

        v1 = F.adaptive_max_pool2d(v10, output_size=(sz, sz))
        v1 = v1.squeeze(1).view(-1, sz**2)

        v1 = F.relu(self.fc_spatial_sq(v1))
        v1 = self.fc_spatial_ex(v1)

        v1 = v1.view(-1, sz, sz).unsqueeze(1)
        v1 = F.adaptive_avg_pool2d(v1, output_size=(v10.shape[-2], v10.shape[-1]))

        v2 = F.adaptive_avg_pool2d(v_in, output_size=(1, 1))
        v2 = F.relu(self.fc_channel_sq(v2))
        v2 = self.fc_channel_ex(v2)
        
        outer_prod =  F.sigmoid(v1.expand(-1, d, -1, -1) * v2.expand(-1, -1, h, w))

        outer_prod = rearrange(outer_prod, 'b d h w -> b h w d')

        new_v = v_interp * (1 + outer_prod)

        vvv = rearrange(new_v,  'b h w d -> b (h w) d')

        attention_logits = self.w(vvv, mask=mask) / self.temperature

        attn = rearrange(attention_logits,  'b (h w) d -> b h w d', h=h, w=w)

        # vals = [map_sq_large, map_sq, map_ex_small, map_ex, attn]
        # titles = ['before 2D squeeze', 'squeezed', 'excited', 'enlarged to orignal res', 'attetnion']

        # plt.figure()
        # for id, val in enumerate(vals):
        #     plt.subplot(3,2,id+1)
        #     plt.imshow(val.squeeze().cpu().detach().numpy())
        #     plt.title(titles[id])
        #     plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('squueze_excite2.png')
        # plt.close()

        attn_selected = select_features(attn, metadata)

        return attn_selected

    def forward(    
        self, v: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, metadata=metadata, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
    


def interpolate_coords(features, metadata):

    feat = features.squeeze(0)
    feat_relu = F.relu(feat)

    all_coords = torch.tensor([tuple([int(d) for d in nm.replace('.jpg', '').split('_')[::-1]]) for nm in metadata['all_coords']])
    
    h, w = metadata['level_y_max'], metadata['level_x_max']

    d = feat.shape[-1]
    new_mat2 = torch.zeros((h, w, d), dtype=feat.dtype, device=feat.device)

    new_mat2[all_coords[:, 0], all_coords[:, 1], :] = feat_relu
    # new_mat2 = rbf_interpolation(feat_relu, all_coords.float(), new_mat2)

    new_mat2 = F.tanh(new_mat2)

    return new_mat2.unsqueeze(0)


def select_features(feature_map, metadata):


    coordinates = torch.tensor([tuple([int(d) for d in nm.replace('.jpg', '').split('_')[::-1]]) for nm in metadata['all_coords']])
    
    B, H, W, D = feature_map.shape
    C = coordinates.shape[0]
    
    # Ensure coordinates are within bounds
    assert (coordinates[:, 0] < H).all() and (coordinates[:, 1] < W).all(), "Coordinates out of bounds"
    
    # Create indices for fancy indexing
    indices = torch.stack([torch.arange(B).unsqueeze(1).expand(B, C), coordinates[:, 0].unsqueeze(0).expand(B, C), coordinates[:, 1].unsqueeze(0).expand(B, C)], dim=-1)
    
    # Use fancy indexing to select features
    selected_features = feature_map[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2], :]
    
    return selected_features


def rbf_interpolation(feat, all_coords, new_mat, sigma=1.0):
    # Calculate pairwise distances
    dist_matrix = torch.cdist(all_coords, all_coords, p=2)
    
    # Compute RBF weights
    rbf_weights = torch.exp(-dist_matrix / (2 * sigma**2))
    
    # Normalize weights
    rbf_weights /= (rbf_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Interpolate feat at all_coords
    interpolated_feat = torch.mm(rbf_weights.detach().cuda(), feat)
    
    # Reshape interpolated_feat and assign to new_mat
    new_mat[:, :, :] = 0  # Clear new_mat
    new_mat[all_coords[:, 0].long(), all_coords[:, 1].long(), :] = interpolated_feat

    return new_mat


def rbf_interpolation_torch(all_coords, feat, new_mat, epsilon=1, function='gaussian'):

    feat = feat.detach().cpu().numpy()

    for channel in range(feat.shape[-1]):

        rbf = Rbf(all_coords[:, 0], all_coords[:, 1], feat[..., channel].detach().cpu().numpy(), epsilon=epsilon, function=function)
        
        xx, yy = np.meshgrid(np.arange(new_mat.shape[0]), np.arange(new_mat.shape[1]))
        new_coords = np.column_stack((xx.ravel(), yy.ravel()))
        
        interpolated_values = rbf(new_coords[:, 0], new_coords[:, 1])
        interpolated_values = torch.tensor(interpolated_values, dtype=new_mat.dtype, device=new_mat.device)
        
        # new_mat.data[:interpolated_values.shape[0], :interpolated_values.shape[1]] = interpolated_values.reshape(new_mat.shape[0], new_mat.shape[1], -1)
        
    return new_mat


def inverse_distance_weighting(all_coords, feat, new_mat, p=2, power=2):
    tree = cKDTree(all_coords)
    indices = tree.query_ball_point(np.array(list(np.ndindex(new_mat.shape[:2]))), r=1)
    
    for idx, neighbors in enumerate(indices):
        if len(neighbors) > 0:
            distances = np.linalg.norm(all_coords[neighbors] - idx, ord=p, axis=1)
            weights = 1 / (distances ** power)
            weights /= np.sum(weights)
            new_mat[tuple(np.unravel_index(idx, new_mat.shape))] = np.sum(feat[neighbors] * weights[:, None], axis=0)
    
    return new_mat


from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q *  self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, :n]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
