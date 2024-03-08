import torch
import torch.nn as nn
import numpy as np
from math import ceil
from timm.models.layers import Mlp, DropPath, trunc_normal_, to_2tuple

class Attention(nn.Module):
    """ Multi-Head Attention
    """
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ TNT Block
    """
    def __init__(
            self,
            dim,
            dim_out,
            num_heads_in=4,
            num_heads_out=4,
<<<<<<< HEAD
            mlp_ratio=2.,
=======
            mlp_ratio=4.,
>>>>>>> origin/master
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(dim)
        self.attn_in = Attention(
            dim,
            dim,
            num_heads=num_heads_in,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        
        self.norm_mlp_in = norm_layer(dim)
        self.mlp_in = Mlp(
            in_features=dim,
            hidden_features=int(dim * 2),
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        
        self.norm1_proj = norm_layer(dim)
        #self.proj = nn.Linear(dim * num_pixel, dim_out, bias=True)

        # Outer transformer
        self.norm_out = norm_layer(dim_out)
        self.attn_out = Attention(
            dim_out,
            dim_out,
            num_heads=num_heads_out,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm_mlp = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, pixel_embed):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        
        return pixel_embed

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck),
                nn.ReLU(),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations
        self.relations_scales = []
        self.subsample_scales = []
        self.fusion_vit = nn.ModuleList()
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass
        
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        
        #print(self.scales)
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        Block(self.img_feature_dim, self.img_feature_dim),
                        Mean(),
                        nn.Linear(self.img_feature_dim*scale, num_bottleneck),
                        #nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        )
            self.fc_fusion_scales += [fc_fusion]
            self.fusion_vit += [Block(num_bottleneck, num_bottleneck,)]
        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input, train=True):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0] , :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0], self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
        act_all = act_scale_1.clone()
        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]
            #for idx in idx_relations_randomsample:
            act_relation_all_list = []
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation_o = act_relation.view(act_relation.size(0), self.scales[scaleID], self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation_o)

                if train==True:
            
                    num = torch.rand(1)[0]#random.random()
                    if num < 0.5 or len(act_relation_all_list)==0:
                        act_relation_all_list.append(act_relation)
                    else:
                        pass
                else:
                    act_relation_all_list.append(act_relation)
            act_relation_all_list = torch.stack(act_relation_all_list, 1)
            act_relation_agg = self.fusion_vit[scaleID](act_relation_all_list)
            act_all = torch.cat((act_all, act_relation_agg.mean(1).unsqueeze(1)), 1)
            #act_all = torch.cat((act_all, act_relation_all.mean(1).unsqueeze(1)), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
class Mean(torch.nn.Module):
    def forward(self, x):
        batch_size,t,c = x.shape
        return x.reshape(batch_size, t*c)