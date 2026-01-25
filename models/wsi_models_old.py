import torch.nn as nn
from models.mil_utils.abmil import *
from models.mil_utils.transmil import TransMIL
from models.mil_utils.grasp import GRASP
from models.mil_utils.zoommil import ZoomMIL
from models.mil_utils.attention import MaxPooler
from models.mil_utils.DeepAttnMISL_CS_MIL import DeepAttnMIL_Surv
import models.loralib as lora
import torch
from einops import rearrange


# class Combined_WSI_model(nn.Module):

#     def __init__(self, encoder, out_features=1, mil_model = 'abmil'):
#         super().__init__()
        
#         self.encoder=encoder
#         lora.mark_only_lora_as_trainable(self.encoder)

#         if mil_model.lower() == 'abmil':

#             self.agg_and_classify = ABMIL(in_features=self.encoder.embed_dim, 
#                                 out_features=out_features, 
#                                 d_model_attention=128,
#                                 attn_type='MaxPool')
            
#         elif mil_model.lower() == 'csmil':

#             self.agg_and_classify = DeepAttnMIL_Surv(in_features=self.encoder.embed_dim, 
#                                 out_features=out_features, 
#                                 cluster_num=1,
#                                 d_model_attention=128,
#                                 attn_type='MaxPool')

        
#         print('trainable params in total... :')
#         for n, p in self.named_parameters():
#             if p.requires_grad:
#                 print(n)
        
#     def forward(self, x, lengths):

#         x = self.encoder(x)
#         x = torch.split(x, lengths)

#         if self.training:
#             x = torch.stack(x)
#             outs = self.agg_and_classify(x, -1)
#         else:
#             outs = torch.cat([self.agg_and_classify(xx.unsqueeze(0), -1) for _, xx in enumerate(x)], dim=0)

#         return outs.squeeze(1)
    


class Combined_WSI_model_multi_lora(nn.Module):

    def __init__(self, encoders, out_features=1):
        super().__init__()
        
        self.encoders = nn.ModuleList(encoders)
        num_segments = len(encoders)

        for encoder in self.encoders:
            lora.mark_only_lora_as_trainable(encoder)

        self.agg_and_classify = ABMILv2(in_features=self.encoders[0].embed_dim, 
                                        num_segments=num_segments,
                            out_features=out_features, 
                            d_model_attention=128)
        
        print('trainable params in total... :')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.numel())

        return

    
    def forward_encoders(self, x, segment_lengths):

        x_split = torch.split(x, segment_lengths, dim=1)

        outs = []
        for enc_idx, encoder in enumerate(self.encoders):
            o1 = encoder(rearrange(x_split[enc_idx], 'b n c h w -> (b n) c h w'))
            outs.append(rearrange(o1, '(b n) d -> b n d', b=x.shape[0]))

        return outs
    
        
    def forward(self, x, lengths, segment_lengths):

        x = self.forward_encoders(x, segment_lengths)

        outs = self.agg_and_classify(x)

        return outs.squeeze(1)
    



class Combined_WSI_model_aligned_lora_old(nn.Module):

    def __init__(self,  encoders, out_features=1, mil_config='maxpool', only_lora=True):
        super().__init__()
        
        self.encoders = nn.ModuleList(encoders)
        num_segments = len(encoders)

        if only_lora:
            for encoder in self.encoders:
                lora.mark_only_lora_as_trainable(encoder)

        # self.agg_and_classify = MIL_model('vit_small', 
        #                         out_features=out_features, 
        #                         num_segments=num_segments,
        #                         mil_config=mil_config)

        self.agg_and_classify = DeepAttnMIL_Surv(in_features=encoder.embed_dim, 
                                out_features=out_features, 
                                cluster_num=1, num_mags=num_segments)
        
        return

    
    def forward_encoders(self, x, segment_lengths):

        if len(x.shape) != 6:
            x = x.unsqueeze(0)
        
        b,n,m,c,h,w = x.shape

        if segment_lengths is None:
            segment_lengths = -1 * torch.ones(b,n,m)

        seg_mask = segment_lengths==-1

        output = torch.zeros(b, n, m, self.encoders[0].embed_dim).cuda()

        for enc_idx, encoder in enumerate(self.encoders):

            if seg_mask[:,:,enc_idx].sum() == b*n:
                x_to_encode = x[:, :, enc_idx]
                o1 = encoder(rearrange(x_to_encode, 'b n c h w -> (b n) c h w'))

                output[:, :, enc_idx] = rearrange(o1, '(b n) d -> b n d', b=b)

            else:

                a = segment_lengths[:,:,enc_idx].long()

                unique_masks = (a == -1)

                unique_idxs_all = [torch.nonzero(um).squeeze() for um in unique_masks]

                unique_images = x[unique_masks][:, enc_idx]

                all_vectors = encoder(unique_images)

                enc_vecs = torch.split(all_vectors, [len(u) for u in unique_idxs_all])

                for i, (encoded_vectors, unique_indices) in enumerate(zip(enc_vecs, unique_idxs_all)):
                    output[i, unique_indices, enc_idx] = encoded_vectors
                    for j in range(n):  # For each image in the batch
                        if a[i, j] != -1:
                            output[i, j, enc_idx] = output[i, a[i, j], enc_idx]

        return output
    
        
    def forward(self, x, lengths, segment_lengths):

        x_out = self.forward_encoders(x, segment_lengths)
        outs = self.agg_and_classify(x_out).squeeze(1)

        return outs
 
    

class Combined_WSI_model(nn.Module):

    def __init__(self, encoders, out_features=1, mil_config='maxpool', only_lora=True):
        super().__init__()
        
        self.encoders = nn.ModuleList(encoders)
        num_segments = len(encoders)

        if only_lora:
            for encoder in self.encoders:
                lora.mark_only_lora_as_trainable(encoder)

        self.agg_and_classify = MIL_model('vit_small', 
                                          out_features=out_features, 
                                          num_segments=num_segments,
                                          mil_config=mil_config)
        
        return
    
    
    def forward_encoders(self, x):

        if len(x.shape) != 6:
            x = x.unsqueeze(0)
        
        b,n,m,*img_dims = x.shape

        output = torch.zeros(b, n, m, self.encoders[0].embed_dim).cuda()
        for enc_idx, encoder in enumerate(self.encoders):
            x_enc = x[:, :, enc_idx]
            x_enc = rearrange(x_enc, 'b n c h w -> (b n) c h w')
            out = encoder(x_enc)
            output[:, :, enc_idx] = rearrange(out, '(b n) d -> b n d', b=b)

        return output
    
        
    def forward(self, x):

        x_out = self.forward_encoders(x)

        outs = self.agg_and_classify(x_out)

        return outs
    

    

class Combined_WSI_model_finetune(nn.Module):

    def __init__(self, encoders, out_features=1, mil_config='maxpool'):
        super().__init__()
        
        self.encoder = encoders

        self.agg_and_classify = MIL_model('vit_small', 
                                          out_features=out_features, 
                                          num_segments=1,
                                          mil_config=mil_config)
        
        return

    
    def forward_encoders(self, x):

        if len(x.shape) != 6:
            x = x.unsqueeze(0)
        
        b,n,m,*_= x.shape

        x = rearrange(x, 'b n m ... -> (b n m) ...')

        y = self.encoder(x)

        output = rearrange(y, '(b n m) ... -> b n m ...', b=b, n=n, m=m)

        return output
    
        
    def forward(self, x):

        x_out = self.forward_encoders(x)

        outs = self.agg_and_classify(x_out)

        return outs
    



class Combined_WSI_model_multi_default(nn.Module):

    def __init__(self, encoder, out_features=1, num_segments=3):
        super().__init__()
        
        self.encoder = encoder
        lora.mark_only_lora_as_trainable(encoder)

        self.agg_and_classify = ABMILv2(in_features=self.encoder.embed_dim, 
                                        num_segments=num_segments,
                            out_features=out_features, 
                            d_model_attention=128)

        return

    
    def forward_encoders(self, x, segment_lengths):

        b, n, *_ = x.shape

        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.encoder(x)
        x = rearrange(x, '(b n) d -> b n d', b=b, n=n)

        outs = torch.split(x, segment_lengths, dim=1)

        return outs
    
        
    def forward(self, x, lengths, segment_lengths):

        x = self.forward_encoders(x, segment_lengths)

        outs = self.agg_and_classify(x)

        return outs
    


class Combined_WSI_model_aligned_default(nn.Module):

    def __init__(self, encoder, out_features=1, num_segments=3):
        super().__init__()
        
        self.encoder = encoder
        lora.mark_only_lora_as_trainable(encoder)

        self.agg_and_classify = ABMILv2(in_features=self.encoder.embed_dim, 
                                        num_segments=num_segments,
                            out_features=out_features, 
                            d_model_attention=128)
        
        print('trainable params in total... :')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.numel())

        return

    
    def forward_encoders(self, x, segment_lengths):

        b, n, *_ = x.shape

        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.encoder(x)
        x = rearrange(x, '(b n) d -> b n d', b=b, n=n)

        outs = torch.split(x, segment_lengths, dim=1)

        return outs
    
        
    def forward(self, x, lengths, segment_lengths):

        x = self.forward_encoders(x, segment_lengths)

        outs = self.agg_and_classify(x)

        return outs.squeeze(1)
    


class MIL_model(nn.Module):

    def __init__(self, encoder, out_features=1, num_segments=3, mil_config='csmil', num_subimgs_train=128, num_subimgs_val=-1):
        super().__init__()

        if 'vit_small' in  encoder:
            self.embed_dim = 384
        elif 'swin' in encoder:
            self.embed_dim = 768
        elif 'vit_base' in encoder:
            self.embed_dim = 768
        
        self.encoder = encoder

        if 'csmil' in mil_config:
            cluster_num = 1 if mil_config=='csmil' else 8
            self.agg_and_classify = DeepAttnMIL_Surv(in_features=self.embed_dim, 
                                    out_features=out_features, 
                                    cluster_num=cluster_num, num_mags=num_segments)
            
        elif 'abmil' in mil_config:
            self.agg_and_classify = ABMIL(in_features=self.embed_dim, 
                                               out_features=1, hidden_dim=128, 
                                               use_tiles_embed=True)
            
        elif 'transmil' in mil_config:
            self.agg_and_classify = TransMIL(input_dim=self.embed_dim, n_classes=1, 
                                             hidden_dim=128)
            
        elif 'grasp' in mil_config:
            self.agg_and_classify = GRASP(in_features=self.embed_dim, out_features=1, 
                                             hidden_dim=128, num_subimgs=num_subimgs_train, 
                                             num_subimgs_val=num_subimgs_val, num_mags=num_segments)
            
        elif 'maxpool' in mil_config or 'meanpool' in mil_config:
            use_tiles_embed=False if 'only' in mil_config else True
            self.agg_and_classify = PoolMIL(in_features=self.embed_dim,
                                out_features=out_features, 
                                hidden_dim=128, use_tiles_embed=use_tiles_embed, mil_config=mil_config)
            
        elif 'zoommil' in mil_config:
            self.agg_and_classify = ZoomMIL(in_feat_dim=self.embed_dim, hidden_feat_dim=128,
                                             out_feat_dim=256, n_cls=1)
                    
        # print('trainable params in total... :')
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.numel())

        return
        
    def forward(self, x):

        outs = self.agg_and_classify(x)

        return outs.squeeze(1)