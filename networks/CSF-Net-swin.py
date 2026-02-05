# 替换用：恢复 Swin + 修复 cat 尺寸对齐
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numbers
from einops import rearrange

# 保留你已有的 Swin 导入（请确保此文件在路径中）
from swin_transformer_list import SwinTransformer

# -------------------- 保留的基础模块（略去注释） --------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, max(1, in_channels // reduction))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, in_channels // reduction), in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc1(y); y = self.relu(y); y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class Conv1(nn.Module):
    def __init__(self, n_feat):
        super(Conv1, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=1)
        self.mag = nn.Sequential(nn.BatchNorm2d(n_feat), nn.LeakyReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, 1), nn.BatchNorm2d(n_feat), nn.LeakyReLU(inplace=True))
        self.pha = nn.Sequential(nn.BatchNorm2d(n_feat), nn.LeakyReLU(),
                                 nn.Conv2d(n_feat, n_feat, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, 1), nn.BatchNorm2d(n_feat), nn.LeakyReLU(inplace=True))
        self.pha_se = SEBlock(n_feat)
        self.mag_se = SEBlock(n_feat)
    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre); pha = torch.angle(fre)
        mag_out = self.mag_se(self.mag(mag)); pha_out = self.pha_se(self.pha(pha))
        real = mag_out * torch.cos(pha_out); imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return y + self.main(x)

def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]; return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = max(1, int(dim * ffn_expansion_factor))
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = max(1, num_heads)
        self.qkv2 = nn.Conv2d(dim // 2, dim * 3 // 2, 1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, 3, 1, 1, groups=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim // 2, dim // 2, 1, bias=bias)
        self.avgpool2 = nn.AdaptiveAvgPool2d((4,4))
        self.fourier = Conv1(dim // 2)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = self.fourier(x1)
        b, c, h, w = x2.shape
        qkv = self.qkv_dwconv2(self.qkv2(x2)); q, k, v = qkv.chunk(3, dim=1)
        k = self.avgpool2(k); v = self.avgpool2(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (k.transpose(-2,-1) @ q).softmax(dim=-2)
        out2 = (v @ attn)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return torch.cat([out, self.project_out2(out2)], dim=1)

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

class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x1, x2):
        return self.fuse(torch.cat([x1, x2], dim=1))

class LiteResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(LiteResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=1, bias=True)
        self.channel_shuffle = lambda x: x
        self.attn = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) if in_channels != out_channels else None
    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.act(self.conv1(x)); out = self.conv2(out)
        out = self.channel_shuffle(out)
        out = out + identity
        out = out * torch.sigmoid(self.attn(out))
        return out

class Conv2(nn.Module):
    def __init__(self, subnet_constructor, channel_num, split_len1, clamp=0.8):
        super(Conv2, self).__init__()
        self.split_len1 = split_len1; self.split_len2 = channel_num - split_len1; self.clamp = clamp
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        self.invconv = nn.Conv2d(channel_num, channel_num, 1, 1, 0, groups=1, bias=False)
    def forward(self, x):
        x = self.invconv(x)
        x1, x2 = x[:, :self.split_len1], x[:, self.split_len1:]
        y1 = x1 + self.F(x2)
        s = self.clamp * torch.tanh(self.H(y1))
        y2 = x2 * torch.exp(s) + self.G(y1)
        return torch.cat([y1, y2], dim=1)

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super(FusionBlock, self).__init__()
        self.fuse = nn.Sequential(Conv2(LiteResBlock, 2 * channels, channels), nn.Conv2d(2 * channels, channels, 1))
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1); return self.fuse(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    def forward(self, x): return self.proj(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, 3, 2, 1), nn.BatchNorm2d(n_feat*2), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.down_sample(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(n_feat, max(1, n_feat//2), 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(max(1, n_feat//2)), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.up_sample(x)

class BranchAttention(nn.Module):
    def __init__(self, channels):
        super(BranchAttention, self).__init__()
        hidden = max(8, channels // 8)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
    def forward(self, feat_list):
        b, c, h, w = feat_list[0].shape
        scores = []
        for x in feat_list:
            v = F.adaptive_avg_pool2d(x, 1).view(b, c)
            s = self.mlp(v); scores.append(s)
        scores = torch.stack(scores, dim=1)
        weights = torch.softmax(scores, dim=1)
        fused = weights[:,0].view(b,1,1,1) * feat_list[0] + \
                weights[:,1].view(b,1,1,1) * feat_list[1] + \
                weights[:,2].view(b,1,1,1) * feat_list[2]
        return fused, weights

# -------------------- 主模型：恢复 Swin 并修复尺寸 --------------------
class HAformerSpatialFrequency(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1,1,1,1],
                 heads=[1,1,2,4],
                 ffn_expansion_factor=2.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        super(HAformerSpatialFrequency, self).__init__()

        # stems
        self.stem_art_pv = OverlapPatchEmbed(3, dim)
        self.stem_art = OverlapPatchEmbed(1, dim)
        self.stem_pv = OverlapPatchEmbed(1, dim)
        self.stem_delay = OverlapPatchEmbed(1, dim)

        # BranchAtt + Fusion modules
        self.branch_att_level1 = BranchAttention(dim)
        self.branch_att_level2 = BranchAttention(dim*2)
        self.branch_att_level3 = BranchAttention(dim*4)
        self.branch_att_level4 = BranchAttention(dim*8)
        self.fusion_module1 = FusionBlock(dim)
        self.fusion_module2 = FusionBlock(dim*2)
        self.fusion_module3 = FusionBlock(dim*4)
        self.fusion_module4 = FusionBlock(dim*8)

        # aux conv blocks
        self.conv_block_art1 = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_pv1  = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_delay1 = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_art2 = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_pv2  = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_delay2 = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_art3 = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_pv3  = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_delay3 = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_art4 = UNetConvBlock(dim*8, dim*8, stride=1)
        self.conv_block_pv4  = UNetConvBlock(dim*8, dim*8, stride=1)
        self.conv_block_delay4 = UNetConvBlock(dim*8, dim*8, stride=1)

        # encoder transformer blocks (精简)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
                                              for _ in range(num_blocks[0])])
        self.down_ap_1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
                                              for _ in range(num_blocks[1])])
        self.down_ap_2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
                                              for _ in range(num_blocks[2])])
        self.down_ap_3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                                                       ffn_expansion_factor=ffn_expansion_factor,
                                                       bias=bias, LayerNorm_type=LayerNorm_type)
                                      for _ in range(num_blocks[3])])

        # aux downsampling
        self.downsample_aux1 = UNetConvBlock(dim, dim*2)
        self.downsample_aux2 = UNetConvBlock(dim*2, dim*4)
        self.downsample_aux3 = UNetConvBlock(dim*4, dim*8)

        # Upsample modules
        self.up1 = Upsample(dim*2)
        self.up2 = Upsample(dim*4)
        self.up3 = Upsample(dim*8)

        # convout expects concat channels up1(+),up2(+),up3(+),fused_level1 total dim*8
        self.convout = nn.Conv2d(dim*8, out_channels, 3, 1, 1, bias=bias)

        # ----- 恢复 Swin backbone（轻量化） -----
        # 小型 Swin：embed_dim=48, depths 较小 -> stage 输出 channels: [48,96,192,384]
        self.backbone = SwinTransformer(
            img_size=224, patch_size=4, in_chans=dim, num_classes=0,
            embed_dim=48, depths=[2,2,2,1], num_heads=[3,6,12,24],
            window_size=7, ape=False, drop_path_rate=0.1, patch_norm=True,
        )
        # 投影到当前 dim
        self.swin_proj1 = nn.Conv2d(96, dim, 1)
        self.swin_proj2 = nn.Conv2d(192, dim*2, 1)
        self.swin_proj3 = nn.Conv2d(384, dim*4, 1)
        self.swin_proj4 = nn.Conv2d(384, dim*8, 1)
        self.swin_norm1 = nn.BatchNorm2d(dim)
        self.swin_norm2 = nn.BatchNorm2d(dim*2)
        self.swin_norm3 = nn.BatchNorm2d(dim*4)
        self.swin_norm4 = nn.BatchNorm2d(dim*8)

    def _reshape_swin_feat(self, x, target_hw=None):
        # x: [B, N, C] or [B, C, H, W]
        if x is None:
            return None
        if x.ndim == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        if target_hw is not None:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x

    def forward(self, art, pv, delay):
        # art,pv,delay: single-channel each for your setup; we concatenate to 3-channel input for stem
        art_pv = torch.cat([art, pv, delay], dim=1)  # [B,3,H,W]
        feat_art_pv = self.stem_art_pv(art_pv)       # [B, dim, H, W]

        # Swin takes in channels = dim (we set backbone in_chans=dim), so feed feat_art_pv
        swin_feats = self.backbone.forward_features(feat_art_pv)  # list of 4 stages: each [B, N, C_stage]

        # reshape + project
        swin1 = self._reshape_swin_feat(swin_feats[0], feat_art_pv.shape[-2:])
        swin1 = self.swin_proj1(swin1); swin1 = self.swin_norm1(swin1)
        swin2 = self._reshape_swin_feat(swin_feats[1], None); swin2 = self.swin_proj2(swin2); swin2 = self.swin_norm2(swin2)
        swin3 = self._reshape_swin_feat(swin_feats[2], None); swin3 = self.swin_proj3(swin3); swin3 = self.swin_norm3(swin3)
        swin4 = self._reshape_swin_feat(swin_feats[3], None); swin4 = self.swin_proj4(swin4); swin4 = self.swin_norm4(swin4)

        # auxiliary stems
        feat_art = self.stem_art(art); feat_pv = self.stem_pv(pv); feat_dl = self.stem_delay(delay)

        # ---------- Stage1 ----------
        out_enc_level1_main = self.encoder_level1(feat_art_pv + swin1)
        out_art_l1 = self.conv_block_art1(feat_art); out_pv_l1 = self.conv_block_pv1(feat_pv); out_dl_l1 = self.conv_block_delay1(feat_dl)
        fused_aux_l1, _ = self.branch_att_level1([out_art_l1, out_pv_l1, out_dl_l1])
        fused_level1 = self.fusion_module1(out_enc_level1_main, fused_aux_l1)

        art_pv_downsampled1 = self.down_ap_1_2(fused_level1 + out_enc_level1_main)
        aux_downsampled1 = self.downsample_aux1(fused_level1 + fused_aux_l1)

        # ---------- Stage2 ----------
        # align swin2 spatial to art_pv_downsampled1
        swin2 = self._reshape_swin_feat(swin2, art_pv_downsampled1.shape[-2:])
        out_enc_level2_main = self.encoder_level2(art_pv_downsampled1 + swin2)
        out_art_l2 = self.conv_block_art2(aux_downsampled1); out_pv_l2 = self.conv_block_pv2(aux_downsampled1); out_dl_l2 = self.conv_block_delay2(aux_downsampled1)
        fused_aux_l2, _ = self.branch_att_level2([out_art_l2, out_pv_l2, out_dl_l2])
        fused_level2 = self.fusion_module2(out_enc_level2_main, fused_aux_l2)

        art_pv_downsampled2 = self.down_ap_2_3(fused_level2 + out_enc_level2_main)
        aux_downsampled2 = self.downsample_aux2(fused_level2 + fused_aux_l2)

        # ---------- Stage3 ----------
        swin3 = self._reshape_swin_feat(swin3, art_pv_downsampled2.shape[-2:])
        out_enc_level3_main = self.encoder_level3(art_pv_downsampled2 + swin3)
        out_art_l3 = self.conv_block_art3(aux_downsampled2); out_pv_l3 = self.conv_block_pv3(aux_downsampled2); out_dl_l3 = self.conv_block_delay3(aux_downsampled2)
        fused_aux_l3, _ = self.branch_att_level3([out_art_l3, out_pv_l3, out_dl_l3])
        fused_level3 = self.fusion_module3(out_enc_level3_main, fused_aux_l3)

        art_pv_downsampled3 = self.down_ap_3_4(fused_level3 + out_enc_level3_main)
        aux_downsampled3 = self.downsample_aux3(fused_level3 + fused_aux_l3)

        # ---------- Stage4 ----------
        swin4 = self._reshape_swin_feat(swin4, art_pv_downsampled3.shape[-2:])
        out_enc_level4_main = self.latent(art_pv_downsampled3 + swin4)
        out_art_l4 = self.conv_block_art4(aux_downsampled3); out_pv_l4 = self.conv_block_pv4(aux_downsampled3); out_dl_l4 = self.conv_block_delay4(aux_downsampled3)
        fused_aux_l4, _ = self.branch_att_level4([out_art_l4, out_pv_l4, out_dl_l4])
        fusion_level4 = self.fusion_module4(out_enc_level4_main, fused_aux_l4)

        # ---------- Upsample ----------
        up1 = self.up1(fused_level2)    # might be smaller spatial
        up2 = self.up2(fused_level3)
        up3 = self.up3(fusion_level4)

        # --- 关键：对齐空间尺寸（以 fused_level1 为基准） ---
        target_size = fused_level1.shape[2:],  # (H, W)
        target_hw = fused_level1.size()[2:]
        up1 = F.interpolate(up1, size=target_hw, mode='bilinear', align_corners=False)
        up2 = F.interpolate(up2, size=target_hw, mode='bilinear', align_corners=False)
        up3 = F.interpolate(up3, size=target_hw, mode='bilinear', align_corners=False)

        # 再次确认 shape 一致（可选 debug）
        # print("shapes:", up1.shape, up2.shape, up3.shape, fused_level1.shape)

        output = torch.cat([up1, up2, up3, fused_level1], dim=1)
        output = self.convout(output)
        return output

# 权重初始化与测试
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1); init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.02)
        if m.bias is not None: init.constant_(m.bias, 0)

if __name__ == "__main__":
    from thop import profile
    model = HAformerSpatialFrequency()
    model.apply(weights_init)
    art = torch.rand((1,1,224,224))
    pv = torch.rand((1,1,224,224))
    dl = torch.rand((1,1,224,224))
    flops, params = profile(model, inputs=(art, pv, dl))
    print(f"FLOPs: {flops/1e9:.4f} G")
    print(f"Params: {params/1e6:.4f} M")
