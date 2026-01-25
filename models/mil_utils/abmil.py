import torch.nn as nn
import torch

from typing import List, Optional
from models.mil_utils.attention import *
from models.mil_utils.mlp import MLP
from models.mil_utils.tile_layers import TilesMLP


class ABMIL_old(nn.Module):
    """Attention-based MIL classification model (See [1]_).

    Example:
        >>> module = ABMIL(in_features=128, out_features=1)
        >>> logits, attention_scores = module(slide, mask=mask)
        >>> attention_scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    d_model_attention: int = 128
        Dimension of attention scores.
    temperature: float = 1.0
        GatedAttention softmax temperature.
    tiles_mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in first MLP.
    mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in last MLP.
    mlp_dropout: Optional[List[float]] = None,
        Dropout rate for last MLP.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation for last MLP.
    bias: bool = True
        Add bias to the first MLP.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based
    deep multiple instance learning. In Jennifer Dy and Andreas Krause,
    editors, Proceedings of the 35th International Conference on Machine
    Learning, volume 80 of Proceedings of Machine Learning Research,
    pages 2127–2136. PMLR, 10–15 Jul 2018.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 0,
        attn_type='GatedAttention',
    ) -> None:
        super(ABMIL, self).__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length"
                    "as mlp_dropout if mlp_dropout is given."
                )

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )

        self.attention_layer = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )


        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.metadata_cols = metadata_cols

    def score_model(
        self, x: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, metadata, mask)
        return attention_logits

    def forward(
        self, features: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, D+3)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        tiles_emb = self.tiles_emb(features[..., self.metadata_cols:], mask)
        scaled_tiles_emb, _ = self.attention_layer(tiles_emb, metadata, mask)
        logits = self.mlp(scaled_tiles_emb)

        return logits
    




class ABMILv2(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 0,
        num_segments=1
    ) -> None:
        super().__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length"
                    "as mlp_dropout if mlp_dropout is given."
                )

        self.tiles_embs = nn.ModuleList([TilesMLP(in_features,hidden=tiles_mlp_hidden,bias=bias,out_features=d_model_attention) for _ in range(num_segments)])

        self.attention_layers = nn.ModuleList([MaxPooler(d_model=d_model_attention, temperature=temperature) for _ in range(num_segments)])

        mlp_in_features = d_model_attention * num_segments

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.metadata_cols = metadata_cols

    def score_model(
        self, x: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, metadata, mask)
        return attention_logits

    def forward(
        self, features: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        


        tiles_embs = [embedder(feat[..., self.metadata_cols:], mask) for feat, embedder in zip(features, self.tiles_embs)]
        scaled_embs = [pooler(tiles_emb, metadata, mask)[0] for tiles_emb, pooler in zip(tiles_embs, self.attention_layers)]

        scaled_embs = torch.cat(scaled_embs, dim=-1)
        
        logits = self.mlp(scaled_embs)

        return logits
    


class ABMILv3(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 0,
        num_segments=1
    ) -> None:
        super().__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length"
                    "as mlp_dropout if mlp_dropout is given."
                )

        self.tiles_embs = nn.ModuleList([TilesMLP(in_features,hidden=tiles_mlp_hidden,bias=bias,out_features=d_model_attention) for _ in range(num_segments)])

        self.attention_layers = nn.ModuleList([MaxPooler(d_model=d_model_attention, temperature=temperature) for _ in range(num_segments)])

        mlp_in_features = d_model_attention * num_segments

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.metadata_cols = metadata_cols

    def score_model(
        self, x: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, metadata, mask)
        return attention_logits

    def forward(
        self, features: torch.Tensor, metadata=None, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        


        tiles_embs = [embedder(feat[..., self.metadata_cols:], mask) for feat, embedder in zip(features, self.tiles_embs)]
        scaled_embs = [pooler(tiles_emb, metadata, mask)[0] for tiles_emb, pooler in zip(tiles_embs, self.attention_layers)]

        scaled_embs = torch.cat(scaled_embs, dim=-1)
        
        logits = self.mlp(scaled_embs)

        return logits
    
import re

class WeibullDensityActivation(nn.Module):
    def __init__(self, a=1.0, b=1.0):
        super(WeibullDensityActivation, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        # Apply the Weibull PDF formula

        part1 = self.a / (self.b ** self.a)
        part2 = x  ** (self.a - 1)
        part3 = torch.exp(-((x / self.b) ** self.a))
        return part1 * part2 * part3
    
class PoolMIL(nn.Module):

    def __init__(
        self, in_features, out_features, hidden_dim = 128, use_tiles_embed=True, mil_config='maxpool'
    ):
        super().__init__()

        aa = re.search(r'\d+', mil_config)
        myVals = {'2':102, '3':88, '4':78, '5':72, '6':68}
        
        if use_tiles_embed:
            
            if aa is None:
                self.tiles_embs = nn.Linear(in_features, hidden_dim)
                attn_dim=hidden_dim
            else:
                dim = myVals[str(aa.group())]
                n = int(aa.group()) - 1
                if 'tanh' in mil_config:
                    self.tiles_embs = nn.Sequential(nn.Linear(in_features, dim), 
                                                *[nn.Sequential(nn.Tanh(), nn.Linear(dim, dim)) for _ in range(n)])
                elif 'softsign' in mil_config:
                    self.tiles_embs = nn.Sequential(nn.Linear(in_features, dim), 
                                                *[nn.Sequential(nn.Softsign(), nn.Linear(dim, dim)) for _ in range(n)])
                elif 'sigmoid' in mil_config:
                    self.tiles_embs = nn.Sequential(nn.Linear(in_features, dim), 
                                                *[nn.Sequential(nn.Sigmoid(), nn.Linear(dim, dim)) for _ in range(n)])
                elif 'none' in mil_config:
                    self.tiles_embs = nn.Sequential(nn.Linear(in_features, dim), 
                                                *[nn.Linear(dim, dim) for _ in range(n)])
                else:
                    self.tiles_embs = nn.Sequential(nn.Linear(in_features, dim), 
                                                *[nn.Sequential(nn.ReLU(), nn.Linear(dim, dim)) for _ in range(n)])
                attn_dim=dim

            self.mlp = nn.Linear(attn_dim, out_features)
        else:
            self.tiles_embs = nn.Identity()
            attn_dim= in_features
            self.mlp = nn.Sequential(nn.Linear(in_features, out_features))
            
        if 'max' in mil_config:
            self.attention_layer = MaxPooler()
        else:
            self.attention_layer = MeanPooler()
        # self.mlp = nn.Linear(attn_dim, out_features)

    def forward(self, features):
        
        tiles_embs = self.tiles_embs(features)

        # idx=15

        # torch.save(tiles_embs.detach().cpu(), f'visualizations/sup_inet_features/outputs_intermediate_{idx}.pt')
        # torch.save(tiles_embs.detach().cpu(), f'visualizations/{model_config}/outputs_intermediate_{idx}.pt')
        # # torch.save(tiles_embs.detach().cpu(), f'visualizations/ibot_pan4m_features/outputs_intermediate_{idx}.pt')

        # we want to get the global maxpool for all tiles and mags
        scaled_embs = self.attention_layer(tiles_embs, dim=[1,2]).squeeze(1,2)
        
        logits = self.mlp(scaled_embs)

        return logits
    



class ABMIL(nn.Module):

    def __init__(
        self, in_features, out_features, hidden_dim = 128, use_tiles_embed=True, gate_only=False
    ):
        super().__init__()
        
        if use_tiles_embed:
            self.tiles_embs = nn.Linear(in_features, hidden_dim)
        else:
            self.tiles_embs = nn.Identity()

        self.attention_layer =  GatedAttention(d_model=hidden_dim, temperature=1.0, gate_only=gate_only)
        self.mlp = nn.Linear(hidden_dim, out_features)

    def forward(self, features):

        # input is in the form  (B, SEQ_LEN, M, IN_FEATURES)
        
        tiles_embs = self.tiles_embs(features)

        # we need to make the shape in the form: (B, SEQ_LEN, IN_FEATURES). This is only valid for single mag as I need results quick!

        tiles_embs = rearrange(tiles_embs, 'b s m d -> b (s m) d')
        
        scaled_embs, attns = self.attention_layer(tiles_embs)
        logits = self.mlp(scaled_embs)

        return logits
    