"""
TODO: verify that my global init functions doesn't change init of PoolingLayers
TODO: think whether drop-path on trnsformer layers should include pooling-layers too. (Currently yes.)
TODO: think about positional encodings:
    - if they are indeed just for position, then no need to have new position embeddings at every
      new level
    - but if they are there to promote variations among different blocks of a same level, then thery
      may be needed at every level again. The fact the NesT re-uses a new position embedding at
      every new level suggests the latter (since positions were already encoded uniquely for every
      patch at the bottom layer).
        - probably not needed: I get better results without the additionnal position-embeddings at
          every layer. (This may also come from the fact that, thanks to the upsampling
          convolutions, there is already some spatial info encoded.
"""
import collections.abc
import logging
import math
from functools import partial
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
from .helpers import build_model_with_cfg, named_apply
from .layers import PatchEmbed, Mlp, DropPath, create_classifier, trunc_normal_
from .layers import create_conv2d, create_pool2d, to_ntuple
from .registry import register_model

from einops import rearrange as ra

_logger = logging.getLogger(__name__)

def _cfg_cifar(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': [4, 4],
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': CIFAR10_DEFAULT_MEAN, 'std': CIFAR10_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'multiresolution': _cfg_cifar()
}

class Pooling(nn.Module):
    def __init__(self, dim, pooltype):
        '''
        num_levels: number of scales in the hierarchy
        dim: number of dimensions in attention layer
        '''
        super().__init__()
        pooltypes = pooltype.split('-')
        layers = []
        for name in pooltypes:
            if name == 'conv3':
                layers.append(
                    nn.Conv2d(dim, dim, kernel_size=3, stride=2, groups=dim, bias=False))
            elif name in {'conv', 'conv2'}:
                layers.append(
                    nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=False))
            elif name in {'mp3', 'maxpool3'}:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            elif name in {'maxpool', 'maxpool2', 'mp', 'mp2'}:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif name in {'ln', 'norm', 'layernorm'}:
                layers.append(nn.LayerNorm(dim))
            else:
                raise NotImplementedError(f'pooltype {pooltype} unknown')
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x.shape expected to be B C H W
        """
        for layer in self.layers:
            if type(layer) == nn.LayerNorm:
                x = layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = layer(x)
        return x  # B C H W


class PoolingLayers(nn.Module):
    def __init__(self, num_levels, dim, pooltype='conv3-ln-maxpool3'):
        """
        num_levels: number of scales in the hierarchy
        dim: number of dimensions in attention layer
        pooltype: combination of 'conv', 'conv3', 'ln', 'maxpool', 'maxpool3' separated by '-'
        """
        super().__init__()
        self.num_levels = num_levels
        self.poolings = nn.ModuleList([Pooling(dim, pooltype) for _ in range(num_levels-1)])

    def forward(self, x, level=0):
        if len(x.shape) == 5:
            B, T, H, W, C = x.shape
            bH = bW = int(math.sqrt(T))  # block-hight/width (assumes original image was square)
            x = ra(x, 'B (bH bW) H W C -> B (bH H) (bW W) C', bH=bH, bW=bW)
        else:
            T = None
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        for i in range(self.num_levels-1-level):
            x = self.poolings[i](x)
        x = x.permute(0, 2, 3, 1)  # B C h w -> B h w C
        if T is not None:
            x = ra(x, 'B (bH h) (bW w) C -> B (bH bW) h w C', bH=bH, bW=bW)
        return x


class ScaledAttention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """
    def __init__(self, level, poolings, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.level = level  # level of this attention module (0 = largest scale)
        self.poolings = poolings  # B C H W -> B C h w
        self.qk = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B T H W C (bottom-level embedded image dimensions)
        Rmk:
            Each block consists of several patches, each patch consists of several pixels. I.e., a
            patch is a group of pixels that has been downscaled/pooled to the current level/scale.
            K denotes the number of heads
        """ 
        z = self.poolings(x, self.level)
        B, T, H, W, C = x.shape  # B T C (h k) (w l)
        B, T, h, w, C = z.shape

        z = ra(z, 'B T h w C -> B T (h w) C')  # B T P C
        x = ra(x, 'B T (h i) (w j) C -> B T (i j) (h w) C', h=h, w=w)  # B T N P C
        N = x.shape[2]  # nbr of pixels per patch (i.e., of high-res pixels per low-res pixel)
        P = z.shape[2]  # nbr of patches (at current level) per block = h*w

        v = ra(self.v(x), 'B T N P (K c) -> B K T N P c', K=self.num_heads)  # c = C // num_heads
        qk = ra(self.qk(z), 'B T P (a K c) -> a B K T P c', a=2, K=self.num_heads)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B K T P P
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn[:, :, :, None, :, :]  # B K T 1 P P

        # (B, K, T, N, P, c), permute -> (B, T, N, P, c, K)
        x = ra(attn @ v, 'B K T N P c -> B T N P (K c)')  # B T N P C
        x = self.proj(x)  # TODO: why do we need projection here, if we continue with MLP anyway?
        x = self.proj_drop(x)
        x = ra(x, 'B T (i j) (h w) C -> B T (h i) (w j) C', h=h, i=H//h)  # B T H W C
        return x


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses ScaledAttention layer that handles the "block" dimension
    """
    def __init__(self, level, poolings, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_layer=ScaledAttention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(level, poolings, dim, num_heads=num_heads,
                               qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        x.shape = B T H W C
        """
        y = self.norm1(x)  # TODO: maybe move to inside ScaledAttention, after downscaling/pooling
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerLevel(nn.Module):
    """ Single hierarchical level of TopDown
    """
    def __init__(
            self, level, depth, poolings, embed_dim, num_heads,
            mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rates=[],
            act_layer=None, norm_layer=None, attn_layer=None, pad_type=''):
        super().__init__()

        self.level = level

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_layers = nn.Sequential(*[
            TransformerLayer(
                level=level, poolings=poolings, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i],
                act_layer=act_layer, norm_layer=norm_layer, attn_layer=attn_layer)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, H, W, C) (no blocks here)
        """
        B, H, W, C = x.shape
        Th = Tw = int(2 ** self.level)

        # blockify
        x = ra(x, 'B (Th h) (Tw w) C -> B (Th Tw) h w C', Th=Th, Tw=Tw)

        # layers
        x = self.transformer_layers(x)  # (B, T, H', W', C)

        # deblockify
        x = ra(x, 'B (Th Tw) h w C -> B (Th h) (Tw w) C', Th=Th, Tw=Tw)  # B H W C
        return x


class TopDown(nn.Module):
    def __init__(self, img_size=32, in_chans=3, patch_size=1, num_levels=3, embed_dim=192,
                 num_heads=(3, 3, 3), depths=(1, 1, 1), num_classes=10, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5, norm_layer=None, act_layer=None,
                 attn_layer=None, pad_type='', weight_init='', global_pool='avg'):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size (S in paper)
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dim (int): embedding dimensions of each level (must be int!)
            num_heads (int, tuple): number of attention heads for each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map

        Notes:
            - `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - `embed_dim` must be an int (contrary to NesT)
        """
        super().__init__()

        for param_name in ['num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'

        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.feature_info = []
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        attn_layer = attn_layer or ScaledAttention
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], 'Model only handles square inputs'
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size

        # Number of blocks at bottom level
        num_blocks = (4 ** torch.arange(num_levels)).tolist()
        assert (img_size // patch_size) % math.sqrt(num_blocks[-1]) == 0, \
            'Bottom level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`'

        # Block edge size in units of patches (at current level)
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(num_blocks) is the
        # number of blocks along edge of image
        block_size = int((img_size // patch_size) // math.sqrt(num_blocks[-1]))
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.poolings = PoolingLayers(num_levels, embed_dim)

        # Build up each hierarchical level
        downlevels, uplevels = [], []  # TODO: test: downlevels = uplevels
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = 1
        for l in range(num_levels):
            args = [l, depths[l], self.poolings, embed_dim, num_heads[l], mlp_ratio, qkv_bias, drop_rate,
                    attn_drop_rate, dp_rates[l], act_layer, norm_layer, attn_layer, pad_type]
            downlevels.append(TransformerLevel(*args))
            uplevels.append(TransformerLevel(*args))
            self.feature_info += [dict(num_chs=embed_dim, reduction=curr_stride, module=f'levels.{l}')]
            curr_stride *= 2
        uplevels.reverse()
        self.downlevels = nn.Sequential(*downlevels)
        self.uplevels = nn.Sequential(*uplevels[::-1])

        # Final normalization layer
        self.norm = norm_layer(embed_dim)

        # Classifier
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # for level in chain(self.downlevels, self.uplevels):  # no pos_embed on TransformerLevel
        #     trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}  # TODO: fix this
        # return {f'level.{i}.pos_embed' for i in range(len(self.levels))}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.patch_embed(x)  # B C H W
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x + self.pos_embed
        x = self.downlevels(x)  # B H W C
        x = self.uplevels(x)  # B H W C
        x = self.poolings(x, level=0)  # B H W C
        x = self.norm(x)  # TODO: test exchanging this and previous lines
        return x.permute(0,3,1,2)  # -> B C H W

    def forward(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.forward_features(x) # B C H W
        x = self.global_pool(x)  # B C  # TODO: do we really want to average over all (H W)?
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.head(x)


def _init_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        if 'poolings' in name:
            _, c, kH, kW = module.weight.shape
            nn.init.constant_(module.weight, 1./(c*kH*kW))
            print(f'Initializing pooling conv layer in {name} with weight {1./(c*kH*kW)}')
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

# def resize_pos_embed(posemb, posemb_new):
#     """
#     Rescale the grid of position embeddings when loading from state_dict
#     Expected shape of position embeddings is (1, T, N, C), and considers only square images
#     """
#     _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
#     seq_length_old = posemb.shape[2]
#     num_blocks_new, seq_length_new = posemb_new.shape[1:3]
#     size_new = int(math.sqrt(num_blocks_new*seq_length_new))
#     # First change to (1, C, H, W)
#     posemb = deblockify(posemb, int(math.sqrt(seq_length_old))).permute(0, 3, 1, 2)
#     posemb = F.interpolate(posemb, size=[size_new, size_new], mode='bicubic', align_corners=False)
#     # Now change to new (1, T, N, C)
#     posemb = blockify(posemb.permute(0, 2, 3, 1), int(math.sqrt(seq_length_new)))
#     return posemb
# 
# 
# def checkpoint_filter_fn(state_dict, model):
#     """ resize positional embeddings of pretrained weights """
#     pos_embed_keys = [k for k in state_dict.keys() if k.startswith('pos_embed_')]
#     for k in pos_embed_keys:
#         if state_dict[k].shape != getattr(model, k).shape:
#             state_dict[k] = resize_pos_embed(state_dict[k], getattr(model, k))
#     return state_dict


def _create_nest(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(
        TopDown, variant, pretrained,
        default_cfg=default_cfg,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        # pretrained_filter_fn=checkpoint_filter_fn,  # TODO: write positional embedding function
        **kwargs)

    return model


@register_model
def multiresolution(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=32, patch_size=1, num_levels=3, embed_dim=192, num_heads=3, depths=1,
        num_classes=10, **kwargs)
    model = _create_nest('multiresolution', pretrained=pretrained, **model_kwargs)
    return model

# @register_model
# def multiresolution(pretrained=False, **kwargs):
#     model = TopDown()
#     return model
