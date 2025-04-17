## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_base(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_base, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_base(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_base, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_AFS(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_AFS, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        x = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        x = self.shift(x)
        return x

    def ifft(self, x, n=128):
        x = self.unshift(x)
        x = torch.fft.ifft2(x, norm='forward', dim=(-2,-1))
        x = torch.abs(x)
        return x

    def createMask(self, x, ratio = 0.2):
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        for i in range(mask.shape[0]):
            h_ = int((h * ratio))
            w_ = int((w * ratio))

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
        return mask

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # q = self.fft(q)
        # k = self.fft(k)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.fft(attn)

        mask1 = self.createMask(attn, ratio=0)
        attn1 = attn * (1 - mask1)
        attn1 = self.ifft(attn1)

        mask2 = self.createMask(attn, ratio=0.1)
        attn2 = attn * (1 - mask2)
        attn2 = self.ifft(attn2)

        mask3 = self.createMask(attn, ratio=0.2)
        attn3 = attn * (1 - mask3)
        attn3 = self.ifft(attn3)

        mask4 = self.createMask(attn, ratio=0.3)
        attn4 = attn * (1 - mask4)
        attn4 = self.ifft(attn4)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, AFS_Flag = False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if AFS_Flag:
            self.attn = Attention_AFS(dim, num_heads, bias)
            self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        else:
            self.attn = Attention_base(dim, num_heads, bias)
            self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class DKG(nn.Module):

    def __init__(self, in_channels, kernel_size=3):
        super(DKG, self).__init__()
        self.kernel_size = kernel_size
        self.dynamic_conv = nn.Sequential(
            nn.Linear(512, in_channels * kernel_size * kernel_size),
            nn.ReLU(),
            nn.Linear(in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, F_d, F_c):
        B, C, H, W = F_d.shape
        kernel_weights = self.dynamic_conv(F_c)  # (B, C*K*K)
        kernel_weights = kernel_weights.view(B, C, self.kernel_size, self.kernel_size)  # (B, C, K, K)
        
        pad = self.kernel_size // 2
        F_d_padded = F.pad(F_d, (pad, pad, pad, pad), mode='reflect')
        F_out = torch.zeros_like(F_d)

        for i in range(B):
            F_out[i] = F.conv2d(F_d_padded[i:i+1], kernel_weights[i].unsqueeze(1), groups=C)

        return F_out

class CDFA(nn.Module):
    def __init__(self, in_channels, degradation_dim=512, num_heads=8, expansion_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.degradation_dim = degradation_dim
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.degradation_proj = nn.Sequential(
            nn.Linear(degradation_dim, in_channels * 2),
            nn.GELU()
        )
        
        self.spatial_conv = DKG(in_channels, 3)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        hidden_dim = int(in_channels * expansion_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, F_d, F_c):
        B, C, H, W = F_d.shape
        
        # DKG
        spatial_feat = self.spatial_conv(F_d, F_c)  # [B,C,H,W]
        
        # DQA
        degradation_params = self.degradation_proj(F_c)  # [B, 2C]
        gamma, beta = degradation_params.chunk(2, dim=1)  # [B,C] each
        
        query = gamma.view(B, 1, C)  # [B,1,C]
        key_value = spatial_feat.flatten(2).permute(0,2,1)  # [B,HW,C]
        
        attn_feat, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False
        )  # [B,1,C]
        
        attn_feat = attn_feat.view(B, C, 1, 1) * beta.view(B, C, 1, 1)
        attn_feat = attn_feat.expand(-1, -1, H, W)
        
        # GFEF
        fused_feat = F_d + attn_feat
        enhanced_feat = self.ffn(fused_feat)
        gate = self.fusion_gate(enhanced_feat)  # [B,2,1,1]
        F_aligned = gate[:,:1] * enhanced_feat + gate[:,1:] * fused_feat
        
        return F_aligned

##########################################################################
##---------- Restormer -----------------------
class DEUIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(DEUIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=False) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=False) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=False) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=False) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=True) for i in range(num_blocks[2])])

        self.CA_level3 = CDFA(int(dim*2**2), 512, heads[1], ffn_expansion_factor)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=True) for i in range(num_blocks[1])])

        self.CA_level2 = CDFA(int(dim*2**1), 512, heads[0], ffn_expansion_factor)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=True) for i in range(num_blocks[0])])

        self.CA_level1 = CDFA(int(dim*2**1), 512, heads[0], ffn_expansion_factor)

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, AFS_Flag=False) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
        # CLIP Fusion Feature (not used)
        self.conv_ClIP_level1 = nn.Conv2d(512, int(dim*2**1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def fill_ClIP_Prompt(self, degra_features, latent):
        B, C = degra_features.shape
        _, _, H, W = latent.shape
        p = degra_features.view(B, C, 1, 1)
        p = F.interpolate(p, size=(H, W))
        return p

    def forward(self, inp_img, degra_features=None):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # CLIP Fusion Feature
        if degra_features is not None:
            inp_dec_level3 = self.CA_level3(inp_dec_level3, degra_features)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # CLIP Fusion Feature
        if degra_features is not None:
            inp_dec_level2 = self.CA_level2(inp_dec_level2, degra_features)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # CLIP Fusion Feature
        if degra_features is not None:
            inp_dec_level1 = self.CA_level1(inp_dec_level1, degra_features)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1