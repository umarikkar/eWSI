import torch.nn as nn
from models.mil_utils.abmil import *
from models.mil_utils.acmil import ACMIL_GA, ACMIL_GA_LinMax
from models.mil_utils.transmil import TransMIL
from models.mil_utils.grasp import GRASP
from models.mil_utils.zoommil import ZoomMIL
from models.mil_utils.attention import MaxPooler
from models.mil_utils.DeepAttnMISL_CS_MIL import DeepAttnMIL_Surv
import models.loralib as lora
import torch
from einops import rearrange
from models.mil_utils.snuffy import SnuffyClass


class Combined_WSI_model(nn.Module):

    def __init__(self, encoders, out_features=1, mil_config='maxpool', lora_config='default', num_segments=1, encoder_name='vit_base'):
        super().__init__()
        
        if lora_config=='features':
            self.forward_fn = self.forward_identity
        else:
            self.forward_fn = self.forward_features
            self.encoders = nn.ModuleList(encoders)
            if 'lora' in lora_config or 'default' in lora_config or 'dora' in lora_config:
                for encoder in self.encoders:
                    lora.mark_only_lora_as_trainable(encoder)

        # torch.manual_seed(1)
        self.mil_model = MIL_model(encoder_name, 
                                out_features=out_features, 
                                num_segments=num_segments,
                                mil_config=mil_config)
        
        
        return


    def forward_features(self, x):

        if len(x.shape) != 6:
            x = x.unsqueeze(0)
        
        b,n,m,*_ = x.shape

        output = torch.zeros(b, n, m, self.encoders[0].embed_dim).cuda()
        for enc_idx, encoder in enumerate(self.encoders):
            x_enc = x[:, :, enc_idx]
            x_enc = rearrange(x_enc, 'b n ... -> (b n) ...')
            out = encoder(x_enc)
            output[:, :, enc_idx] = rearrange(out, '(b n) ... -> b n ...', b=b)

        return output
    

    def forward_identity(self, x):

        return x


    def forward(self, x):

        x_out = self.forward_fn(x)
        outs = self.mil_model(x_out)

        return outs
    

class MIL_model(nn.Module):

    def __init__(self, encoder_name, out_features=1, num_segments=3, mil_config='csmil', num_subimgs_train=128, num_subimgs_val=-1):
        super().__init__()

        if 'vit_small' in  encoder_name:
            self.embed_dim = 384
        elif 'swin' in encoder_name:
            self.embed_dim = 768
        elif 'vit_base' in encoder_name:
            self.embed_dim = 768
        elif 'resnet50' in encoder_name:
            self.embed_dim = 2048

        hidden_dim=128
        use_tiles_embed=True

        if 'only' in mil_config:
            hidden_dim=self.embed_dim
            use_tiles_embed=False
            
        if 'pca' in encoder_name or 'lin' in encoder_name:
            self.embed_dim=hidden_dim=128

        if 'reduced' in mil_config:
            hidden_dim=88

        gate_only=True if 'gate' in mil_config else False
        
        if 'abmil' in mil_config:
            self.agg_and_classify = ABMIL(in_features=self.embed_dim, 
                                               out_features=out_features, hidden_dim=hidden_dim, 
                                               use_tiles_embed=use_tiles_embed, gate_only=gate_only)
            
        elif 'linmax_acmil' in mil_config:
            self.agg_and_classify = ACMIL_GA_LinMax(D_feat=self.embed_dim,
                                             D_inner=88,
                                             n_class=out_features,
                                             conf_n_token=5,
                                            D=128, droprate=0, n_token=5, n_masked_patch=10, mask_drop=0.6,
                                             use_tiles_embed=use_tiles_embed, gate_only=gate_only)
            
        elif 'acmil' in mil_config:
            self.agg_and_classify = ACMIL_GA(D_feat=self.embed_dim,
                                             D_inner=hidden_dim,
                                             n_class=out_features,
                                             conf_n_token=5,
                                            D=128, droprate=0, n_token=5, n_masked_patch=10, mask_drop=0.6,
                                             use_tiles_embed=use_tiles_embed, gate_only=gate_only)
        elif 'snuffy' in mil_config:
            self.agg_and_classify = SnuffyClass(feats_size=self.embed_dim, num_classes=out_features,)

        elif 'transmil' in mil_config:
            self.agg_and_classify = TransMIL(input_dim=self.embed_dim, n_classes=out_features, 
                                             hidden_dim=hidden_dim, use_tiles_embed=use_tiles_embed)

        elif 'maxpool' in mil_config or 'meanpool' in mil_config:
            use_tiles_embed=False if 'only' in mil_config else True
            self.agg_and_classify = PoolMIL(in_features=self.embed_dim,
                                out_features=out_features, 
                                hidden_dim=128, use_tiles_embed=use_tiles_embed, mil_config=mil_config)
            
        return
        
    def forward(self, x):

        outs = self.agg_and_classify(x)

        if type(outs)!=tuple:
            return outs.squeeze(1)
        else:
            return outs