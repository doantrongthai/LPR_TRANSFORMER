"""
Multi-frame CRNN — cải tiến accuracy:
  Backbone  : EfficientNet-B0 pretrained (thay CNNBackbone)
  Fusion    : CrossAttentionFusion frame-wise (thay AttentionFusion)
  Sequence  : GRU → Transformer hybrid (thay BiLSTM thuần)
  Loss      : CTC chính + auxiliary CTC sau GRU (deep supervision)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import CrossAttentionFusion, EfficientNetBackbone, STNBlock


class MultiFrameCRNN(nn.Module):
    """
    Improved CRNN for multi-frame license plate OCR.
    Pipeline:
        [B, F, 3, H, W]
        → STN (per frame)
        → EfficientNet-B0 backbone (pretrained, stride modified)
        → CrossAttentionFusion (frame-wise cross-attention)
        → GRU (local pattern)
        → Transformer encoder (global context)
        → CTC head  (+  auxiliary CTC head sau GRU)
    """

    NUM_FRAMES = 5

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 1024,
        dropout: float = 0.1,
        rnn_dropout: float = None,   # alias cho dropout, giữ tương thích config cũ
        use_stn: bool = True,
    ):
        super().__init__()
        if rnn_dropout is not None:
            dropout = rnn_dropout
        self.cnn_channels = 512
        self.hidden_size  = hidden_size
        self.use_stn      = use_stn

        # ── 1. STN ───────────────────────────────────────────────
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # ── 2. Backbone: EfficientNet-B0 pretrained ──────────────
        self.backbone = EfficientNetBackbone(out_channels=self.cnn_channels)

        # ── 3. Fusion: frame-wise cross-attention ────────────────
        self.fusion = CrossAttentionFusion(
            channels=self.cnn_channels,
            num_frames=self.NUM_FRAMES,
            num_heads=4,
            dropout=dropout,
        )

        # ── 4a. GRU — local pattern ──────────────────────────────
        self.gru = nn.GRU(
            input_size=self.cnn_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # ── 4b. Auxiliary CTC head (sau GRU, deep supervision) ───
        self.aux_head = nn.Linear(hidden_size * 2, num_classes)

        # ── 4c. Transformer encoder — global context ─────────────
        self.input_proj = nn.Linear(hidden_size * 2, self.cnn_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN ổn định hơn
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.norm = nn.LayerNorm(self.cnn_channels)

        # ── 5. Main CTC head ─────────────────────────────────────
        self.head = nn.Linear(self.cnn_channels, num_classes)

    # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, F, 3, H, W]
        Returns (training):
            main_logits : [B, W', num_classes]   log_softmax
            aux_logits  : [B, W', num_classes]   log_softmax
        Returns (eval):
            main_logits : [B, W', num_classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)           # [B*F, 3, H, W]

        # ── STN ──────────────────────────────────────────────────
        if self.use_stn:
            theta     = self.stn(x_flat)           # [B*F, 2, 3]
            grid      = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_flat    = F.grid_sample(x_flat, grid, align_corners=False)

        # ── Backbone ─────────────────────────────────────────────
        features = self.backbone(x_flat)           # [B*F, 512, 1, W']

        # ── Fusion ───────────────────────────────────────────────
        fused = self.fusion(features)              # [B, 512, 1, W']

        # ── GRU ──────────────────────────────────────────────────
        seq = fused.squeeze(2).permute(0, 2, 1)   # [B, W', 512]
        gru_out, _ = self.gru(seq)                # [B, W', hidden*2]

        # Auxiliary output (deep supervision)
        aux_out = self.aux_head(gru_out).log_softmax(2)   # [B, W', C]

        # ── Transformer ──────────────────────────────────────────
        t_in  = self.input_proj(gru_out)          # [B, W', 512]
        t_out = self.transformer(t_in)            # [B, W', 512]
        t_out = self.norm(t_out)

        # ── Main head ────────────────────────────────────────────
        main_out = self.head(t_out).log_softmax(2)        # [B, W', C]

        # Lưu aux_out vào attribute để trainer lấy nếu cần
        # Luôn trả về 1 tensor để tương thích trainer
        self.aux_logits = aux_out
        return main_out