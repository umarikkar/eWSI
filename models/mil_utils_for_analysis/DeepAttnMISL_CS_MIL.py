"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""

import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

class DeepAttnMIL_Surv(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, in_features, out_features, cluster_num=1, num_mags=3):
        super(DeepAttnMIL_Surv, self).__init__()
        self.embedding_net = nn.Sequential(nn.Conv2d(in_features, 64, 1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((num_mags,1))
                                     )

        self.res_attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),  # V
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, out_features),
            # nn.Sigmoid()
        )
        self.cluster_num = cluster_num



    def forward(self, x):

        " x is a tensor and it should be of shape N, D, M, 1 so we will reshape it to that"

        if len(x.shape)==4:
            #  we create a single mini-bag
            x = x.unsqueeze(1) 

        b, n, c, m, d = x.shape
        x = rearrange(x, 'b n c (m m2) d -> (b n c) d m m2' , m=m)

        x = self.embedding_net(x)
        res_attention = self.res_attention(x).squeeze(-1)
        final_output = torch.bmm(x.squeeze(-1), torch.transpose(res_attention,2,1)).squeeze(-1)

        h = rearrange(final_output, '(b n c) d -> (b n) c d', b=b, c=c)

        A = self.attention(h)
        A = torch.transpose(A, 2, 1) 
        A = F.softmax(A, -1)  # '(b n) 1 c'

        M = torch.bmm(A, h)  # # '(b n) 1 d'

        Y_pred = self.fc6(M).squeeze(1) # '(b n) 1'

        # if have_clusters:
        Y_pred = rearrange(Y_pred, '(b n) d -> b n d', b=b)
        Y_pred = Y_pred.mean(1) # 'b 1'
        

        return Y_pred

