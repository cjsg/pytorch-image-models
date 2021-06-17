import logging
from math import prod
from einops import rearrange, repeat as ra, repeat

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .registry import register_model
    from .layers import DropPath
except ImportError:
    from registry import register_model
    from layers import DropPath


_logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

'''
    b = batch_size
    k = num_heads
    h/w = hight/width of image at scale i
    h1/w1 = hight/width of image at scale i+1 = h/w_blocks at scale i
    h2/w2 = hight/width of a block at scale i
    h3/w3 = hight/width of a block at scale i-1 = nbr of h/w_kids of each word at scale i = h_kids

    TODOs:
        - implement weight initialization
        - implement re-weighting of attention (beware: must depend on the block-sizes)
        - implement option to use same weights at all scales
        - implement different queries/keys for parents/peers/kids
        - implement overlapping attention fields to kids (might be difficult)
        - implement recurrent traversal of hierarchy (instead of parallel computations)
'''

# Usual layers

class AddLearnableConstants(nn.Module):
    # This class just adds a learnable constant. It is only here to avoid using nn.ParameterList in
    # the MSEmbedding class, since that would be incompatible with nn.DataParallel.
    def __init__(self, *dims):
        super().__init__()
        self.constants = nn.Parameter(torch.zeros(*dims))

    def forward(self, x):
        return x + self.constants


# class QKV(nn.Module):
#     def __init__(self, dim, num_heads, bias=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.qkv = nn.Linear(dim, 3*dim, bias=bias)
# 
#     def forward(self, x):
#         # dim(x) = b x h x w x feature_dim
#         qkv = self.qkv(x)
#         q, k, v = ra(qkv, 'b h w (i k f) -> i b k h w f', i=3, k=self.num_heads)
#         return q, k, v

class Query(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        # dim(x) = b x h x w x feature_dim
        q = self.q(x)
        q = ra(q, 'b h w (k f) -> b k h w f', k=self.num_heads)
        return q

class KeyValue(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.kv = nn.Linear(in_dim, 2*out_dim, bias=bias)

    def forward(self, x):
        # dim(x) = b x h x w x feature_dim
        kv = self.kv(x)
        k, v = ra(kv, 'b h w (i k f) -> i b k h w f', i=2, k=self.num_heads)
        return k, v


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop),)

    def forward(self, z):
        return self.mlp(z)


# Make layers/tensor multi-scale

class MSTensor(list):
    def __add__(self, ms_y):
        assert len(self) == len(ms_y), (
            f'Both MSTensor must have same length, but {len(self)} and {len(ms_y)} given.')
        ms_z = []
        for (x, y) in zip(self, ms_y):
            ms_z.append(x+y)
        return MSTensor(ms_z)

    def get_scale(self, i):
        return super().__getitem__(i)


class MSLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.ms_layer = nn.ModuleList(layers)

    def forward(self, ms_input):
        ms_out = []
        for x, layer in zip(ms_input, self.ms_layer):
            ms_out.append(layer(x))
        return MSTensor(ms_out)


# Utilities

def compute_scales_and_block_sizes(img_size, patch_size, words_per_block):
    # Compute num_scales, listify words_per_block if needed and compute
    # the num_patches in each block
    num_patches_at_scale_i = img_size // patch_size
    num_patches = [num_patches_at_scale_i]
    if type(words_per_block) == int:
        while num_patches_at_scale_i > 1:
            num_patches_at_scale_i //= words_per_block
            num_patches.append(num_patches_at_scale_i)
        num_patches.reverse()
        num_scales = len(num_patches)
        words_per_block = [1] + [words_per_block] * (num_scales-1)
    else:
        for wpb in reversed(words_per_block[1:]):
            num_patches_at_scale_i //= wpb
            num_patches.append(num_patches_at_scale_i)
        num_patches.reverse()
        num_scales = len(num_patches)

    _logger.debug(f'nbr of scales: {num_scales}')
    _logger.debug(f'words_per_block: {words_per_block}')
    _logger.debug(f'num_patches: {num_patches}')
    assert num_scales == len(num_patches) == len(words_per_block)
    assert prod(words_per_block) * patch_size == img_size, (
        f'words_per_block={words_per_block} patch_size={patch_size} '
        f'img_size={img_size}')

    return num_scales, words_per_block, num_patches


def listify(val_or_list, num_scales):
    if type(val_or_list) != list:
        val_or_list = [val_or_list] * num_scales
    assert len(val_or_list) == num_scales, f'val_or_list: {val_or_list}'
    return val_or_list


# Multiscale ViT

class MSEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, words_per_block,
                 feature_dims, drops=0., num_channels=3):

        super().__init__()
        num_scales, words_per_block, num_patches = compute_scales_and_block_sizes(
            img_size, patch_size, words_per_block)
        self.num_scales = num_scales
        self.words_per_block = words_per_block
        self.num_patches = num_patches
        feature_dims = listify(feature_dims, num_scales)
        drops = listify(drops, num_scales)

        # Make layers
        self.embeddings = nn.ModuleList([
            nn.Conv2d(
                num_channels, dim, kernel_size=patch_size, stride=patch_size)
                    for dim in feature_dims])
        # The most natural way to code the position embedding would be to use a ParameterList, but
        # that is currently incompatible with nn.DataParallel. See
        # https://github.com/pytorch/pytorch/issues/36035 . That is why we resort to this ModuleList
        self.add_pos = nn.ModuleList([
            AddLearnableConstants(h, h, dim)
                for h, dim in zip(num_patches, feature_dims)])
        self.embed_drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])

    def forward(self, x):
        # Decompose x in multi-scale x
        x_cur = x
        ms_x = [x_cur]
        for wpb in reversed(self.words_per_block[1:]):
            x_cur = F.avg_pool2d(x_cur, kernel_size=wpb)
            ms_x.append(x_cur)

        # Embed each scale of x
        ms_out = []
        for i in range(self.num_scales):
            out = self.embeddings[i](ms_x.pop())
            out = ra(out, 'b f h w -> b h w f')
            ms_out.append(self.embed_drops[i](self.add_pos[i](out)))
        return MSTensor(ms_out)


class MSAttention(nn.Module):
    def __init__(self, words_per_block, feature_dims, num_heads, attend_to_peers=True,
                 attend_to_parents=True, decouple_scales=False, qkv_bias=False, attn_drops=0.,
                 proj_drops=0., weight_parents=1., weight_peers=1., weight_kids=1.):
        '''
        words_per_block (list)    list of length `num_scales` containing the number of patches in
                                  each block
        '''
        super().__init__()
        assert type(words_per_block) == list
        self.num_scales = len(words_per_block)
        self.words_per_block = words_per_block
        feature_dims = listify(feature_dims, self.num_scales)
        attn_drops = listify(attn_drops, self.num_scales)
        proj_drops = listify(proj_drops, self.num_scales)
        self.attend_to_peers = attend_to_peers
        self.attend_to_parents = attend_to_parents
        self.decouple_scales = decouple_scales
        self.weight_parents = weight_parents
        self.weight_peers = weight_peers
        self.weight_kids = weight_kids

        if not attend_to_parents and weight_parents not in {0., 1.}:
            raise ValueError('Cannot specify a non-zero attention weight to parents when '
                             '`attend_to_parents` is set to False')
        if not attend_to_peers and weight_peers not in {0., 1.}:
            raise ValueError('Cannot specify a non-zero attention weight to peers when '
                             '`attend_to_peers` is set to False')

        # Remark: num_heads must be int. Cannot be different in every scale yet, given the usual
        # attention mechanism. If we move to a model where each query can be tested against all
        # `num_heads` keys (1 key per head), then we could let the num_heads vary between scales.
        # Then we could also introduce separate values for `num_key_heads` and `num_query_heads`.

        if self.attend_to_parents:
            self.ms_q_par = nn.ModuleList(
                [Query(dim_par, dim_peer, num_heads, qkv_bias)
                    for dim_par, dim_peer in zip(feature_dims[1:], feature_dims[:-1])])
            self.ms_kv_par = nn.ModuleList([
                KeyValue(dim_par, dim_peer, num_heads, qkv_bias)
                    for dim_par, dim_peer in zip(feature_dims[:-1], feature_dims[1:])])

        if self.attend_to_peers:
            self.ms_q_peer = nn.ModuleList([
                Query(dim, dim, num_heads, qkv_bias) for dim in feature_dims])
            self.ms_kv_peer = nn.ModuleList([
                KeyValue(dim, dim, num_heads, qkv_bias) for dim in feature_dims])

        self.ms_q_kid = nn.ModuleList([
            Query(dim_kid, dim_peer, num_heads, qkv_bias)
                for dim_kid, dim_peer in zip(feature_dims[:-1], feature_dims[1:])])
        self.ms_kv_kid = nn.ModuleList([
            KeyValue(dim_kid, dim_peer, num_heads, qkv_bias)
                for dim_kid, dim_peer in zip(feature_dims[1:], feature_dims[:-1])])

        self.attn_drops = [nn.Dropout(attn_drop) for attn_drop in attn_drops]
        self.ms_proj = MSLayer([nn.Linear(dim,dim) for dim in feature_dims])
        self.ms_proj_drop = MSLayer([nn.Dropout(proj_drop) for proj_drop in proj_drops])

    def forward(self, ms_x):
        device = ms_x[0].get_device()
        if device < 0:
            device = 'cpu'
        ms_slf_attn = MSTensor([])

        for i in range(self.num_scales):
            h_block = self.words_per_block[i]
            n_parents = 1 if i > 0 else 0
            h_kids = self.words_per_block[i+1] if i < (self.num_scales-1) else 0
            qk_scaling = (h_block + n_parents + h_kids) ** -0.5 

            attn = []
            val = []
            attn_weights = []

            # Attention to peers
            if self.attend_to_peers:
                qk_scaling_peer = h_block ** -0.5 if self.decouple_scales else qk_scaling
                q_peer = self.ms_q_peer[i](ms_x.get_scale(i))
                k_peer, v_peer = self.ms_kv_peer[i](ms_x.get_scale(i))
                q_peer = ra(q_peer, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                k_peer = ra(k_peer, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                attn.append(torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_peer, k_peer) * qk_scaling_peer)
                v_peer = ra(v_peer, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                val.append(repeat(v_peer, 'b k h1 w1 h2w2 f -> b k h1 w1 r h2w2 f', r=h_block**2))
                attn_weights.append(self.weight_peers * torch.ones(h_block**2, device=device))

            # Attention to parents
            if self.attend_to_parents and i > 0:
                qk_scaling_par = 1. if self.decouple_scales else qk_scaling
                q_par = self.ms_q_par[i-1](ms_x.get_scale(i))  # query constructed from scale i
                k_par, v_par = self.ms_kv_par[i-1](ms_x.get_scale(i-1))  # kv constructed from scale i-1
                q_par = ra(q_par, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                k_par = ra(k_par, 'b k h1 w1 f -> b k h1 w1 1 f')
                attn.append(torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_par, k_par) * qk_scaling_par)
                v_par = ra(v_par, 'b k h1 w1 f -> b k h1 w1 1 f')
                val.append(repeat(v_par, 'b k h1 w1 1 f -> b k h1 w1 r 1 f', r=h_block**2))
                attn_weights.append(self.weight_parents * torch.ones(1, device=device))

            # Attention to kids
            if i < (self.num_scales-1):
                qk_scaling_kids = h_kids ** -0.5 if self.decouple_scales else qk_scaling
                q_kid = self.ms_q_kid[i](ms_x.get_scale(i))  # query constructed from scale i
                k_kid, v_kid = self.ms_kv_kid[i](ms_x.get_scale(i+1))  # kv constructed from scale i+1
                q_kid = ra(q_kid, 'b k h w f -> b k h w 1 f')  # query to kid
                k_kid = ra(k_kid, 'b k (h h3) (w w3) f -> b k h w (h3 w3) f', h3=h_kids, w3=h_kids)
                attn_kid = torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_kid, k_kid) * qk_scaling_kids
                attn.append(ra(attn_kid, 'b k (h1 h2) (w1 w2) 1 g -> b k h1 w1 (h2 w2) g', h2=h_block, w2=h_block))
                val.append(ra(v_kid, 'b k (h1 h2 h3) (w1 w2 w3) f -> b k h1 w1 (h2 w2) (h3 w3) f',
                             h2=h_block, h3=h_kids, w2=h_block, w3=h_kids))
                attn_weights.append(self.weight_kids * torch.ones(h_kids**2, device=device))

            _logger.debug(f'shape of val peers/par/kids: {[v.shape for v in val]}')
            val = torch.cat(val, dim=-2)
            attn_weights = torch.cat(attn_weights, dim=-1)
            if self.decouple_scales:
                for j in range(len(attn)):  # separate weighting of parents, peers and kids
                    attn[j] = attn[j].softmax(dim=-1)
                attn = torch.cat(attn, dim=-1)
            else:
                attn = torch.cat(attn, dim=-1)
                attn = attn.softmax(dim=-1)

            # re-weight attentions
            attn = attn_weights * attn
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # dropout
            attn = self.attn_drops[i](attn)

            # Resulting attn dims
            # attn_peers.shape = b x k x h1 x w1 x h_block**2 x h_block**2
            # attn_paren.shape = b x k x h1 x w1 x h_block**2 x 1
            # attn_kids.shape  = b x k x h1 x w1 x h_block**2 x h_kids**2
            # attn.shape[5] = h_block**2 + n_parents + (h_block*h_kids)**2
            # val.shape = attn.shape x f for all vals

            slf_attn = torch.einsum('bkhwlm,bkhwlnf->bkhwlf', attn, val)
            slf_attn = ra(slf_attn, 'b k h1 w1 (h2 w2) f -> b (h1 h2) (w1 w2) (k f)', h2=h_block, w2=h_block)
            ms_slf_attn.append(slf_attn)

        ms_out = self.ms_proj(ms_slf_attn)
        ms_out = self.ms_proj_drop(ms_slf_attn)
        return ms_out


class MSTransformer(nn.Module):
    def __init__(
            self, words_per_block, feature_dims, num_heads, mlp_hidden_dims, attend_to_peers=True,
            attend_to_parents=True, decouple_scales=False, qkv_bias=False, mlp_drops=0.,
            attn_drops=0., proj_drops=0., path_drops=0., weight_parents=1., weight_peers=1., weight_kids=1.):
        '''
        words_per_block (list)  List of length `num_scales` that starts with
                                value 1
        '''
        super().__init__()

        assert type(words_per_block) == list
        self.num_scales = len(words_per_block)
        self.words_per_block = words_per_block
        feature_dims = listify(feature_dims, self.num_scales)
        mlp_hidden_dims = listify(mlp_hidden_dims, self.num_scales)
        mlp_drops = listify(mlp_drops, self.num_scales)
        attn_drops = listify(attn_drops, self.num_scales)
        proj_drops = listify(proj_drops, self.num_scales)
        path_drops = listify(path_drops, self.num_scales)

        self.ms_ln1 = MSLayer([nn.LayerNorm(dim, eps=1e-6) for dim in feature_dims])
        self.ms_attn = MSAttention(words_per_block, feature_dims, num_heads, attend_to_peers,
                                   attend_to_parents, decouple_scales, qkv_bias, attn_drops,
                                   proj_drops, weight_parents, weight_peers, weight_kids)
        self.ms_ln2 = MSLayer([nn.LayerNorm(dim, eps=1e-6) for dim in feature_dims])
        self.ms_mlp = MSLayer([MLP(dim, hidden_dim, drop=drop)
            for (dim, hidden_dim, drop) in zip(feature_dims, mlp_hidden_dims, mlp_drops)])
        # self.ms_drop_path = MSLayer([DropPath(drop) for drop in path_drops])  # TODO: Solve NaN problem
        self.ms_drop_path = MSLayer([nn.Identity() for drop in path_drops])

    def forward(self, ms_z):
        ms_out = ms_z[:self.num_scales]  # possibly ignore finest scales
        ms_out = self.ms_drop_path(self.ms_attn(self.ms_ln1(ms_z))) + ms_out
        ms_out = self.ms_drop_path(self.ms_mlp(self.ms_ln2(ms_z))) + ms_out
        ms_out.extend(ms_z[self.num_scales:])  # concat ignored scales back
        return ms_out


class MultiScaleViT(nn.Module):
    def __init__(
            self, img_size, num_classes, patch_size, words_per_block, feature_dims, num_heads,
            num_transformers, mlp_hidden_dims, attend_to_peers=True, attend_to_parents=True,
            decouple_scales=False, wait_for_top=False, weight_parents=1., weight_peers=1.,
            weight_kids=1., qkv_bias=False, embed_drops=0., mlp_drops=0., attn_drops=0.,
            proj_drops=0., path_drops=0., num_channels=3):
        '''
        words_per_block (list)  List of length `num_scales` that starts with
                                value 1
        '''
        super().__init__()
        num_scales, words_per_block, num_patches = compute_scales_and_block_sizes(
            img_size, patch_size, words_per_block)

        self.img_size = img_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.words_per_block = words_per_block
        self.num_patches = num_patches
        self.num_transformers = num_transformers
        self.feature_dims = listify(feature_dims, self.num_scales)
        self.num_heads = num_heads
        self.mlp_hidden_dims = listify(mlp_hidden_dims, self.num_scales)
        self.attend_to_peers = attend_to_peers
        self.attend_to_parents = attend_to_parents
        self.decouple_scales = decouple_scales
        self.wait_for_top = wait_for_top
        self.weight_parents = weight_parents
        self.weight_peers = weight_peers
        self.weight_kids = weight_kids
        self.qkv_bias = qkv_bias
        self.embed_drops = listify(embed_drops, self.num_scales)
        self.mlp_drops = listify(mlp_drops, self.num_scales)
        self.attn_drops = listify(attn_drops, self.num_scales)
        self.proj_drops = listify(proj_drops, self.num_scales)
        self.path_drops = listify(path_drops, self.num_scales)
        self.num_channels = num_channels

        self.embedding = MSEmbedding(
            self.img_size, self.patch_size, self.words_per_block,
            self.feature_dims, self.embed_drops, self.num_channels)

        transformers = []
        for l in range(self.num_transformers):
            L = self.num_transformers  # L-l+1 = last small scale layers are irrelevant for computation graph
            s = min(l+2, L-l+1) if self.wait_for_top else L-l+1
            transformers.append(MSTransformer(
                    self.words_per_block[:s], self.feature_dims[:s], self.num_heads,
                    self.mlp_hidden_dims[:s], self.attend_to_peers, self.attend_to_parents,
                    self.decouple_scales, self.qkv_bias, self.mlp_drops[:s], self.attn_drops[:s],
                    self.proj_drops[:s], self.weight_parents, self.weight_peers, self.weight_kids))
        self.transformers = nn.Sequential(*transformers)

        self.head = nn.Linear(self.feature_dims[0], self.num_classes, bias=True)

    def forward(self, x):
        ms_z = self.embedding(x)
        ms_z = self.transformers(ms_z)
        # out = ms_z[0].reshape(x.shape[0], -1)
        out = ms_z.get_scale(0).reshape(x.shape[0], -1)
        return self.head(out)


# beware: this model cannot deal with images of different scales
@register_model
def small_cifar_msvit(pretrained=False, attend_to_peers=True, attend_to_parents=True,
                      decouple_scales=False, wait_for_top=True, weight_parents=1., weight_peers=1.,
                      weight_kids=1., **kwargs):
    dropout=kwargs.pop('drop', .1)
    drop_path=kwargs.pop('drop_path', .1)
    return MultiScaleViT(
                img_size=32,
                num_classes=10,
                patch_size=1,
                words_per_block = [1, 2, 4, 4],
                feature_dims=192,
                num_heads=3,
                num_transformers=12,
                mlp_hidden_dims=4*192,
                attend_to_peers=attend_to_peers,
                attend_to_parents=attend_to_parents,
                decouple_scales=decouple_scales,
                wait_for_top=wait_for_top,
                weight_parents=weight_parents,
                weight_peers=weight_peers,
                weight_kids=weight_kids,
                qkv_bias=False,
                embed_drops=dropout,
                mlp_drops=dropout,
                attn_drops=dropout,
                proj_drops=dropout,
                path_drops=drop_path,
                num_channels=3)


if __name__ == '__main__':
    net = small_cifar_msvit()
    x = torch.randn(2, 3, 32, 32)
    print(net(x))
