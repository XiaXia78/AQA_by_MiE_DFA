from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath

from PoseDetector3d.model.modules.attention import Attention
from PoseDetector3d.model.modules.graph import GCN
from PoseDetector3d.model.modules.mlp import MLP



class STEncoder(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)

        else:
            raise NotImplementedError("mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MiEBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
        super().__init__()
        self.hierarchical = hierarchical
        self.mlp = MLP(in_features=dim, hidden_features=dim * 6, out_features=dim * 3)

        self.att_spatial = STEncoder(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                     qk_scale, use_layer_scale, layer_scale_init_value,
                                     mode='spatial', mixer_type="attention",
                                     use_temporal_similarity=use_temporal_similarity,
                                     neighbour_num=neighbour_num,
                                     n_frames=n_frames)
        self.att_temporal = STEncoder(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                      qk_scale, use_layer_scale, layer_scale_init_value,
                                      mode='temporal', mixer_type="attention",
                                      use_temporal_similarity=use_temporal_similarity,
                                      neighbour_num=neighbour_num,
                                      n_frames=n_frames)

        self.graph = STEncoder(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                               qkv_bias,
                               qk_scale, use_layer_scale, layer_scale_init_value,
                               mode='temporal', mixer_type='graph',
                               use_temporal_similarity=use_temporal_similarity,
                               temporal_connection_len=temporal_connection_len,
                               neighbour_num=neighbour_num,
                               n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        x_spatial = self.att_spatial(x)
        x_temporal = self.att_temporal(x_spatial)
        x_graph = self.graph(x_temporal)
        if self.use_adaptive_fusion:
            alpha = torch.cat((x_graph, x_temporal), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_graph * alpha[..., 0:1] + x_temporal * alpha[..., 1:2]
        else:
            x = x_graph * 0.5 + x_temporal * 0.5
        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
    layers = []
    for _ in range(n_layers):
        layers.append(MiEBlock(dim=dim,
                               mlp_ratio=mlp_ratio,
                               act_layer=act_layer,
                               attn_drop=attn_drop,
                               drop=drop_rate,
                               drop_path=drop_path_rate,
                               num_heads=num_heads,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               qkv_bias=qkv_bias,
                               qk_scale=qkv_scale,
                               use_adaptive_fusion=use_adaptive_fusion,
                               hierarchical=hierarchical,
                               use_temporal_similarity=use_temporal_similarity,
                               temporal_connection_len=temporal_connection_len,
                               use_tcn=use_tcn,
                               graph_only=graph_only,
                               neighbour_num=neighbour_num,
                               n_frames=n_frames))
    layers = nn.Sequential(*layers)
    return layers


class MiEFormer(nn.Module):
    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=27):

        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        x = self.joints_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x
