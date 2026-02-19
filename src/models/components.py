"""Reusable model components for multi-frame OCR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    resnet34,
)


class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8))
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs    = self.localization(x)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        return theta


# ════════════════════════════════════════════════════════════════
# BACKBONE
# ════════════════════════════════════════════════════════════════

class CNNBackbone(nn.Module):
    """Simple CNN backbone for CRNN baseline (giữ nguyên để tương thích)."""
    def __init__(self, out_channels=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, out_channels, 2, 1, 0),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 pretrained backbone, modified for OCR:
      - Giữ stride ngang để bảo toàn sequence width
      - Collapse height về 1 bằng adaptive pool
      - Project về out_channels bằng 1×1 conv
    """
    def __init__(self, out_channels: int = 512, pretrained: bool = True):
        super().__init__()
        weights  = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efnet    = efficientnet_b0(weights=weights)
        features = efnet.features   # Sequential of MBConv blocks

        # ── Chỉnh stride: block 2 và 3 stride (2,1) thay vì (2,2) ──
        # Block index trong EfficientNet-B0 features:
        # 0: stem conv, 1-8: MBConv blocks
        # Giảm stride dọc ở block 3 và 5 để giữ width
        for block_idx in [3, 5]:
            block = features[block_idx]
            for module in block.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (2, 1)
                    # Pad để giữ kích thước
                    module.padding = (
                        module.padding[0],
                        module.padding[1] if module.padding[1] > 0 else 1
                    )
                    break

        self.features = features           # [B, 1280, H', W']

        # Project 1280 → out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(1280, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : [B, 3, H, W]
        Output: [B, out_channels, 1, W']
        """
        x = self.features(x)               # [B, 1280, H', W']
        x = self.proj(x)                   # [B, out_channels, H', W']
        x = F.adaptive_avg_pool2d(x, (1, None))   # [B, out_channels, 1, W']
        return x


# ════════════════════════════════════════════════════════════════
# FUSION
# ════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """Attention-based fusion (giữ nguyên để tương thích với ResTranOCR)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_frames, c, h, w = x.size()
        num_frames = 5
        batch_size = total_frames // num_frames
        x_view  = x.view(batch_size, num_frames, c, h, w)
        scores  = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x_view * weights, dim=1)


class CrossAttentionFusion(nn.Module):
    """
    Frame-wise cross-attention fusion.

    Khác với AttentionFusion (score độc lập per position) và
    TemporalTransformerFusion (flatten F×W' rồi self-attend):
      - Mỗi frame attend TO tất cả frame khác theo frame dimension
      - Sau đó mới aggregate theo spatial dimension
      → Frame yếu học lấy thông tin từ frame mạnh ở đúng vị trí ký tự

    Pipeline:
        [B*F, C, 1, W']
            → reshape [B, W', F, C]       # F frame như 1 sequence per position
            → MultiheadAttention (F → F)  # frame attend to frame
            → weighted sum + residual
            → mean over F
            → [B, C, 1, W']
    """
    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.channels   = channels

        # Cross-attention: query từ mỗi frame, key/value từ tất cả frame
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Variance-based quality prior (không cần learn)
        # Frame ít blur → pixel variance cao → weight cao hơn
        self.quality_proj = nn.Sequential(
            nn.Linear(1, channels),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*F, C, 1, W']
        Returns:
            [B, C, 1, W']
        """
        total, c, h, w = x.size()
        B  = total // self.num_frames
        F  = self.num_frames

        # [B*F, C, 1, W'] → [B, F, C, W'] → [B, W', F, C]
        x_seq = x.squeeze(2)                           # [B*F, C, W']
        x_seq = x_seq.view(B, F, c, w)                # [B, F, C, W']
        x_seq = x_seq.permute(0, 3, 1, 2)             # [B, W', F, C]

        # ── Variance-based quality weight ────────────────────────
        # Tính variance per frame per position → frame rõ hơn weight cao hơn
        var = x_seq.var(dim=-1, keepdim=True)          # [B, W', F, 1]
        quality_weight = self.quality_proj(var)        # [B, W', F, C]

        # Scale features theo quality
        x_weighted = x_seq * quality_weight            # [B, W', F, C]

        # ── Cross-attention giữa frames ───────────────────────────
        # Flatten B và W' để dùng MultiheadAttention: [B*W', F, C]
        bw = B * w
        x_flat = x_weighted.reshape(bw, F, c)         # [B*W', F, C]

        attn_out, _ = self.cross_attn(
            query=x_flat,
            key=x_flat,
            value=x_flat,
        )                                              # [B*W', F, C]

        # Residual + norm
        attn_out = self.norm(x_flat + self.dropout(attn_out))

        # Mean pool over frames → aggregate F frames thành 1
        attn_out = attn_out.mean(dim=1)               # [B*W', C]

        # Reshape về [B, C, 1, W']
        fused = attn_out.view(B, w, c)                # [B, W', C]
        fused = fused.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, W']
        return fused


# ════════════════════════════════════════════════════════════════
# RESNET (giữ nguyên cho ResTranOCR)
# ════════════════════════════════════════════════════════════════

class ResNetFeatureExtractor(nn.Module):
    """ResNet34 backbone cho ResTranOCR (giữ nguyên)."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights    = ResNet34_Weights.DEFAULT if pretrained else None
        resnet     = resnet34(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.layer3[0].conv1.stride         = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        self.layer4[0].conv1.stride         = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, (1, None))


# ════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING + TEMPORAL FUSION (giữ nguyên cho ResTranOCR)
# ════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TemporalTransformerFusion(nn.Module):
    """Temporal Transformer Fusion cho ResTranOCR (giữ nguyên)."""
    def __init__(self, channels, num_frames=5, num_heads=8,
                 num_layers=2, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.channels   = channels
        self.frame_pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1, channels) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total, c, h, w = x.size()
        B = total // self.num_frames
        F = self.num_frames
        x = x.squeeze(2).view(B, F, c, w).permute(0, 1, 3, 2)
        x = x + self.frame_pos_embedding
        x = self.transformer(self.norm(x.reshape(B, F * w, c)))
        x = x.view(B, F, w, c).mean(dim=1)
        return x.permute(0, 2, 1).unsqueeze(2)