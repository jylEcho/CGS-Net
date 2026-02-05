
import sys
import os
# 将当前文件所在目录添加到Python搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
from MSDNN import InvBlock,HinResBlock

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        return x * y


class Conv2(nn.Module):
    def __init__(self, n_feat):
        super(Conv2, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=n_feat)
        self.mag = nn.Sequential(nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
                                 nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(inplace=True),
                                 )
        self.pha = nn.Sequential(nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
                                 nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(inplace=True),
                                 )
        self.pha_se = SEBlock(n_feat)
        self.mag_se = SEBlock(n_feat)

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)

        norm_mag = mag
        norm_pha = pha
        if torch.isnan(norm_mag.any()):
            print('nan')

        mag_out = self.mag_se(self.mag(norm_mag))
        pha_out = self.pha_se(self.pha(norm_pha))

        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return y + self.main(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim // 2, dim * 3 // 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, kernel_size=3, stride=1, padding=1, groups=dim * 3 // 2,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)

        self.qkv2 = nn.Conv2d(dim // 2, dim * 3 // 2, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, kernel_size=3, stride=1, padding=1,
                                     groups=dim * 3 // 2, bias=bias)
        self.project_out2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        self.fourier = Conv2(dim // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)

        out = self.fourier(x1)

        b, c, h, w = x2.shape

        qkv = self.qkv_dwconv2(self.qkv2(x2))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.avgpool2(k)
        v = self.avgpool2(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (k.transpose(-2, -1) @ q)
        attn = attn.softmax(dim=-2)

        out2 = (v @ attn)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out2 = self.project_out2(out2)

        return torch.cat([out, out2], dim=1)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm3(x))

        return x

######################## fusion model ##################################
class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        fused = torch.cat([x1, x2], dim=1)
        fused = self.fuse(fused)
        return fused

# class INNFusion(nn.Module):
#     def __init__(self, channels):
#         super(INNFusion, self).__init__()
#         # 我保持了原来实现
#         self.fuse = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
#                       nn.Conv2d(2 * channels, channels, 1, 1, 0))
#     def forward(self, x1,x2):
#         x = self.fuse(torch.cat([x1,x2],dim=1))
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteResBlock(nn.Module):
    """我改进了一下残差，如果你有更好的设计，可以与我联系"""
    def __init__(self, in_channels, out_channels, groups=2):
        super(LiteResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=True)
        self.channel_shuffle = lambda x: x.view(x.size(0), 2, -1, x.size(2), x.size(3)).transpose(1, 2).reshape_as(x)

        # 轻量注意力
        self.attn = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.channel_shuffle(out)
        out = out + identity
        out = out * torch.sigmoid(self.attn(out))
        return out


class RevFusionBlock(nn.Module):
    """我改进了融合的一个分块"""
    def __init__(self, subnet_constructor, channel_num, split_len1, clamp=0.8):
        super(RevFusionBlock, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = channel_num - split_len1
        self.clamp = clamp

        # 改良残差块代替 HinResBlock
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        # 用 group conv + shuffle 升级 1x1 invertible conv
        self.invconv = nn.Conv2d(channel_num, channel_num, 1, 1, 0, groups=1, bias=False)

    def forward(self, x):
        # 我改进了通道混合
        x = self.invconv(x)

        x1, x2 = (x[:, :self.split_len1], x[:, self.split_len1:])

        y1 = x1 + self.F(x2)
        s = self.clamp * (torch.tanh(self.H(y1)))
        y2 = x2 * torch.exp(s) + self.G(y1)

        return torch.cat([y1, y2], dim=1)


class INNFusion(nn.Module):
    def __init__(self, channels):
        super(INNFusion, self).__init__()
        self.fuse = nn.Sequential(
            RevFusionBlock(LiteResBlock, 2 * channels, channels),
            nn.Conv2d(2 * channels, channels, 1, 1, 0)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
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
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.down_sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_feat*2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.down_sample(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(n_feat//2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up_sample(x)

##########################################################################
## Branch attention: 对三个分支做全局池化->MLP->softmax 得到三分支权重（batch-wise），然后对特征做加权和
class BranchAttention(nn.Module):
    def __init__(self, channels):
        super(BranchAttention, self).__init__()
        # 简单的 MLP: GAP -> linear hidden -> scalar
        hidden = max(8, channels // 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, feat_list):
        # feat_list: list of 3 tensors, each [b, c, h, w]
        b, c, h, w = feat_list[0].shape
        scores = []
        for x in feat_list:
            # global avg pool -> [b, c]
            v = F.adaptive_avg_pool2d(x, 1).view(b, c)
            s = self.mlp(v)  # [b, 1]
            scores.append(s)
        scores = torch.stack(scores, dim=1)  # [b, 3, 1]
        weights = torch.softmax(scores, dim=1)  # across branches
        # 将 weights 应用到特征图（按 batch 广播）
        fused = weights[:,0].view(b,1,1,1) * feat_list[0] + \
                weights[:,1].view(b,1,1,1) * feat_list[1] + \
                weights[:,2].view(b,1,1,1) * feat_list[2]
        return fused, weights  # 返回融合特征和权重（可选用于 debug / 可视化）


##########################################################################
##---------- Restormer-like with three auxiliary branches -----------------------
class HAformerSpatialFrequency(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 3, 3, 4],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False
                 ):

        super(HAformerSpatialFrequency, self).__init__()

        # 主支路 stem: ART+PV concat  (保持原始主干)
        self.stem_art_pv = OverlapPatchEmbed(2, dim)  # 主支路：concat(art,pv)
        # 辅助三分支各自的 stem（单通道）
        self.stem_art = OverlapPatchEmbed(1, dim)
        self.stem_pv = OverlapPatchEmbed(1, dim)
        self.stem_delay = OverlapPatchEmbed(1, dim)

        # BranchAttention 用于把三个辅助分支在每个级别加权融合
        self.branch_att_level1 = BranchAttention(dim)
        self.branch_att_level2 = BranchAttention(dim * 2)
        self.branch_att_level3 = BranchAttention(dim * 4)
        self.branch_att_level4 = BranchAttention(dim * 8)

        # 为每个下采样级别初始化一个融合模块 (主支路与加权后的辅助合并)
        self.fusion_module1 = INNFusion(dim)
        self.fusion_module2 = INNFusion(dim * 2)
        self.fusion_module3 = INNFusion(dim * 4)
        self.fusion_module4 = INNFusion(dim * 8)

        # UNet风格的卷积块（辅助分支 ART、PV、DL 各自的 level conv block）
        # level1
        self.conv_block_art1 = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_pv1  = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_delay1 = UNetConvBlock(dim, dim, stride=1)
        # level2
        self.conv_block_art2 = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_pv2  = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_delay2 = UNetConvBlock(dim*2, dim*2, stride=1)
        # level3
        self.conv_block_art3 = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_pv3  = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_delay3 = UNetConvBlock(dim*4, dim*4, stride=1)
        # level4
        self.conv_block_art4 = UNetConvBlock(dim * 8, dim * 8, stride=1)
        self.conv_block_pv4  = UNetConvBlock(dim * 8, dim * 8, stride=1)
        self.conv_block_delay4 = UNetConvBlock(dim * 8, dim * 8, stride=1)

        # 下采样（主支路）
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_ap_1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down_ap_2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down_ap_3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # 下采样 for fused auxiliary (我们复用 delay 下采样逻辑作为辅助的下采样)
        self.downsample_aux1 = UNetConvBlock(dim, dim * 2)
        self.downsample_aux2 = UNetConvBlock(dim * 2, dim * 4)
        self.downsample_aux3 = UNetConvBlock(dim * 4, dim * 8)

        # 保持您原有的 up modules（用来合并多尺度特征）
        self.up1 = Up(in_channels=128, out_channels=64, layers=1)
        self.up2 = Up(in_channels=256, out_channels=64, layers=2)
        self.up3 = Up(in_channels=512, out_channels=64, layers=3)

        self.convout = nn.Conv2d(256,3,kernel_size=3, stride=1, padding=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_artpv = nn.Conv2d(2, out_channels, kernel_size=1, bias=bias)

    def forward(self, art, pv, delay):
        # 主支路输入：art+pv concat
        art_pv = torch.cat([art, pv], dim=1)
        feat_art_pv = self.stem_art_pv(art_pv)  # 主支路 feature

        # 三个辅助分支单独 stem
        feat_art = self.stem_art(art)
        feat_pv  = self.stem_pv(pv)
        feat_dl  = self.stem_delay(delay)

    ########## 1 stage #############################################################
        # 主支路 encoder level1
        out_enc_level1_main = self.encoder_level1(feat_art_pv)  # 主支路 level1

        # 辅助三分支在 level1 的卷积块
        out_art_l1 = self.conv_block_art1(feat_art)      # [b, dim, H, W]
        out_pv_l1  = self.conv_block_pv1(feat_pv)
        out_dl_l1  = self.conv_block_delay1(feat_dl)

        # BranchAttention：三个辅助分支加权融合（返回融合特征 + 权重）
        fused_aux_l1, weights_l1 = self.branch_att_level1([out_art_l1, out_pv_l1, out_dl_l1])

        # 主/辅助融合
        fused_level1 = self.fusion_module1(out_enc_level1_main, fused_aux_l1)

        # Downsample 主支路 & 辅助融合结果，进入下一级
        art_pv_downsampled1 = self.down_ap_1_2(fused_level1 + out_enc_level1_main)
        aux_downsampled1 = self.downsample_aux1(fused_level1 + fused_aux_l1)

    ########## 2 stage #############################################################
        # 主支路 encoder level2
        out_enc_level2_main = self.encoder_level2(art_pv_downsampled1)

        # 辅助三分支在 level2 的卷积块（对下采样后的辅助特征进行处理）
        out_art_l2 = self.conv_block_art2(aux_downsampled1)  # 注意：我们对下采样后的 fused_aux 进行分支处理
        out_pv_l2  = self.conv_block_pv2(aux_downsampled1)
        out_dl_l2  = self.conv_block_delay2(aux_downsampled1)

        fused_aux_l2, weights_l2 = self.branch_att_level2([out_art_l2, out_pv_l2, out_dl_l2])

        fused_level2 = self.fusion_module2(out_enc_level2_main, fused_aux_l2)

        art_pv_downsampled2 = self.down_ap_2_3(fused_level2 + out_enc_level2_main)
        aux_downsampled2 = self.downsample_aux2(fused_level2 + fused_aux_l2)

    ########## 3 stage #############################################################
        out_enc_level3_main = self.encoder_level3(art_pv_downsampled2)

        out_art_l3 = self.conv_block_art3(aux_downsampled2)
        out_pv_l3  = self.conv_block_pv3(aux_downsampled2)
        out_dl_l3  = self.conv_block_delay3(aux_downsampled2)

        fused_aux_l3, weights_l3 = self.branch_att_level3([out_art_l3, out_pv_l3, out_dl_l3])

        fused_level3 = self.fusion_module3(out_enc_level3_main, fused_aux_l3)

        art_pv_downsampled3 = self.down_ap_3_4(fused_level3 + out_enc_level3_main)
        aux_downsampled3 = self.downsample_aux3(fused_level3 + fused_aux_l3)

    ########## 4 stage #############################################################
        out_enc_level4_main = self.latent(art_pv_downsampled3)

        out_art_l4 = self.conv_block_art4(aux_downsampled3)
        out_pv_l4  = self.conv_block_pv4(aux_downsampled3)
        out_dl_l4  = self.conv_block_delay4(aux_downsampled3)

        fused_aux_l4, weights_l4 = self.branch_att_level4([out_art_l4, out_pv_l4, out_dl_l4])

        fusion_level4 = self.fusion_module4(out_enc_level4_main, fused_aux_l4)

    ########## upsampling stage #############################################################
        up1 = self.up1(fused_level2)
        up2 = self.up2(fused_level3)
        up3 = self.up3(fusion_level4)
        output = torch.cat([up1, up2, up3, fused_level1], 1)

        output = self.convout(output)
        # print("\n最终输出形状:", output.shape)

        return output

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)

if __name__ == "__main__":
    from thop import profile

    model = HAformerSpatialFrequency().cuda('cuda:0')
    model.apply(weights_init)
    x = torch.rand((1, 1, 224, 224)).cuda('cuda:0')
    flops, params = profile(model, inputs=(x,x[:],x[:]))

    print(f"FLOPs: {flops / 1e9} G FLOPs")
    print(f"模型参数数量: {params / 1e6} M")
