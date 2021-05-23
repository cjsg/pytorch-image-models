import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange, repeat

from .registry import register_model

'''
    l = number of attention layers
    d = dim of latent feature vector
    D = hidden dim of mlp
    h = number of SA heads
    n = number of image patches
    p = patch height / width
    c = number of channels
    k = number of classification classes
    f = dim of query, key and value features (Dh in paper)
    NB: image dim = c * n * p^2

'''

class Embed(nn.Module):

    def __init__(self, d, n, p, c=3, dropout_rate=0.):
        super(Embed, self).__init__()
        self.d, self.n, self.p, self.c = d, n, p, c
        self.embed = nn.Linear(c*p**2, d, bias=False)
        self.cls = nn.parameter.Parameter(torch.randn(d))   # class token
        self.pos = nn.parameter.Parameter(torch.randn(n+1, d))  # position
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2 c)',
                      h2=self.p, w2=self.p)
        cls = repeat(self.cls, 'd -> b 1 d', b=x.size(0))
        out = torch.cat([cls, self.embed(x)], dim=1)  # b x (n+1) x d
        return self.dropout(out + self.pos)

class MSA(nn.Module):

    def __init__(self, n, d, h, dropout_rate=0.):
        # n = nbr of image patches
        # d = patch input dim
        # h = nbr of self attention heads
        super(MSA, self).__init__()
        f = int(d / h)  # dimension of query, key & value vectors ('*f*eatures')
        self.h, self.d, self.f = h, d, f
        self.to_qkv = nn.Linear(d, 3*h*f, bias=False)  # input feature to q,k,v vectors
        self.to_mlp = nn.Sequential(
            nn.Linear(h*f, d, bias=False),  # aggregate attention heads and send to MLP
            nn.Dropout(dropout_rate),)

    def forward(self, z):
        b, n, d, h, f = *z.shape, self.h, self.f  # -> renaming n := n+1
        qkv = self.to_qkv(z)
        q, k, v = rearrange(qkv, 'b n (i h f) -> i b h n f', i=3, h=h)  # 3 x b h n f
        dots = torch.einsum('bhnf,bhmf->bhnm', q, k) / sqrt(f) # here, n=m=n
        attn = dots.softmax(dim=3)  # attention weights: b h n n
        slf_attn = torch.einsum('bhnm,bhmf->bhnf', attn, v)  # b h n f
        slf_attn = rearrange(slf_attn, 'b h n f -> b n (h f)')
        return self.to_mlp(slf_attn)  # b n d


class MLP(nn.Module):

    def __init__(self, d, D, dropout_rate=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, D),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(D, d),
            nn.Dropout(dropout_rate),)

    def forward(self, z):
        return self.mlp(z)

class Transformer(nn.Module):
    def __init__(self, d, D, h, n, p, c=3, dropout_rate=0.):
        super(Transformer, self).__init__()
        self.ln1 = nn.LayerNorm(d)  # lucidrains uses LayerNorm(d)
        self.msa = MSA(n, d, h, dropout_rate)
        self.ln2 = nn.LayerNorm(d)  # lucidrains uses LayerNorm(d)
        self.mlp = MLP(d, D, dropout_rate)

    def forward(self, z):
        # dim(z) = b x (n+1) x d
        z = self.msa(self.ln1(z)) + z
        z = self.mlp(self.ln2(z)) + z
        return z

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=1e-6)

class ViT(nn.Module):
    def __init__(self, l, d, D, h, n, p, c=3, k=10, dropout_rate=0.):
        super(ViT, self).__init__()
        self.n, self.c, self.p = n, c, p
        self.dim_checked = False
        self.embedding = Embed(d, n, p, c)
        self.transformer = nn.Sequential(
            *[Transformer(d, D, h, n, p, c, dropout_rate) for _ in range(l)])
        self.ln = nn.LayerNorm(d)
        self.num_classes = k
        self.to_class_logits = nn.Linear(d, k)
        self.apply(init_weights)

    def forward_features(self, x):
        if not self.dim_checked:
            self.dim_checked = True
            if x[0].view(-1).shape[0] != self.n * self.c * self.p**2:
                raise ValueError('patches * patch_size**2 * channels != image dimension\n'
                    f'image dims = {x.shape[1:]}, (n, p, c) = {self.n, self.p, self.c}')
        z = self.embedding(x)
        z = self.transformer(z)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        return self.to_class_logits(self.ln(z[:,0]))


'''
    l = number of attention layers
    d = dim of latent feature vector
    D = hidden dim of mlp
    h = number of SA heads
    n = number of image patches
    p = patch height / width
    c = number of channels
    k = number of classification classes
    f = dim of query, key and value features (Dh in paper)
    NB: image dim = c * n * p^2
'''

@register_model
def ViT_B16(dropout_rate=0., img_size=384, **kwargs):
    assert img_size % 16 == 0, (f'img_size must be a multiple of 16. {img_size} given.')
    patches = img_size // 16
    return ViT(
        l=12, d=768, D=3072, h=12, n=patches**2, p=16, c=3, k=10, dropout_rate=dropout_rate)  # 768 = 256*3 ; 3072 = 768*4

@register_model
def ViT_L2_H4_P4(dropout_rate=0., **kwargs):
    return ViT(
        l=2, D=4*4*3, h=4, n=8*8, d=4*4*3, p=4, c=3, k=10, dropout_rate=dropout_rate)

@register_model
def ViT_L8_H4_P4(dropout_rate=0.1, **kwargs):
    return ViT(
        l=8, D=4*4*3*4, h=4, n=8*8, d=4*4*3, p=4, c=3, k=10, dropout_rate=dropout_rate)


def test():
    # net = Transformer_L8_H4_P4()
    # y = net(torch.randn(1, 3, 32, 32))
    net = ViT_B16()
    y = net(torch.randn(1, 3, 384, 384))
    print(y)
    print(y.size())

# test()
