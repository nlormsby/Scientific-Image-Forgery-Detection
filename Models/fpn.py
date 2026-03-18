"""
Custom Feature Pyramid Network (FPN) Implementation from Scratch

Built from-scratch for ablation studies, similar to MantraNet approach.
Uses pretrained backbone but builds custom FPN decoder with togglable components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetBackbone(nn.Module):
    """ResNet backbone that extracts multi-scale features"""
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()

        # Load pretrained ResNet
        if pretrained:
            weights = models.get_model_weights(backbone_name).DEFAULT
            backbone = models.get_model(backbone_name, weights=weights)
        else:
            backbone = models.get_model(backbone_name, weights=None)

        # Extract feature extraction layers
        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        self.conv1 = backbone.conv1      # /2, 64 channels
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool  # /4, 64 channels

        self.layer1 = backbone.layer1    # /4, 256 channels (ResNet50)
        self.layer2 = backbone.layer2    # /8, 512 channels
        self.layer3 = backbone.layer3    # /16, 1024 channels
        self.layer4 = backbone.layer4    # /32, 2048 channels

        # Store output channels for each stage
        self.out_channels = self._get_out_channels(backbone_name)

    def _get_out_channels(self, backbone_name):
        """Get output channels for each stage based on backbone"""
        if 'resnet18' in backbone_name or 'resnet34' in backbone_name:
            return [64, 128, 256, 512]
        else:  # resnet50, resnet101, resnet152, resnext
            return [256, 512, 1024, 2048]

    def forward(self, x):
        """Extract multi-scale features"""
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Multi-scale features
        c2 = self.layer1(x)   # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        return [c2, c3, c4, c5]


class FPNDecoder(nn.Module):
    """
    FPN Decoder with ablation controls

    Args:
        encoder_channels: List of encoder output channels [C2, C3, C4, C5]
        pyramid_channels: Number of channels in FPN pyramid (default: 256)
        segmentation_channels: Channels in segmentation head (default: 128)
        num_classes: Number of output classes (default: 1)
        use_lateral: Enable lateral connections from encoder (default: True)
        use_topdown: Enable top-down pathway (default: True)
        num_pyramid_levels: Number of pyramid levels to use (default: 4)
        decoder_depth: Number of conv blocks per pyramid level (default: 1)
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(
        self,
        encoder_channels=[256, 512, 1024, 2048],
        pyramid_channels=256,
        segmentation_channels=128,
        num_classes=1,
        use_lateral=True,
        use_topdown=True,
        num_pyramid_levels=4,
        decoder_depth=1,
        dropout=0.2
    ):
        super().__init__()

        self.use_lateral = use_lateral
        self.use_topdown = use_topdown
        self.num_pyramid_levels = num_pyramid_levels
        self.decoder_depth = decoder_depth

        # Only use the specified number of pyramid levels
        encoder_channels = encoder_channels[-num_pyramid_levels:]

        # Lateral connections (1x1 conv to reduce channels)
        if use_lateral:
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(enc_ch, pyramid_channels, kernel_size=1)
                for enc_ch in encoder_channels
            ])
        else:
            # If no lateral, just process encoder features directly
            self.direct_convs = nn.ModuleList([
                nn.Conv2d(enc_ch, pyramid_channels, kernel_size=1)
                for enc_ch in encoder_channels
            ])

        # Top-down pathway (3x3 conv after upsampling + merging)
        if use_topdown and decoder_depth > 0:
            self.fpn_convs = nn.ModuleList([
                nn.Sequential(*[
                    ConvBlock(pyramid_channels, pyramid_channels)
                    for _ in range(decoder_depth)
                ])
                for _ in range(len(encoder_channels))
            ])

        # Segmentation head
        self.seg_blocks = nn.Sequential(
            ConvBlock(pyramid_channels, segmentation_channels),
            nn.Dropout2d(dropout),
            ConvBlock(segmentation_channels, segmentation_channels),
            nn.Conv2d(segmentation_channels, num_classes, kernel_size=1)
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of [C2, C3, C4, C5] from encoder
        Returns:
            Segmentation mask (same size as input)
        """
        # Take only features we need based on num_pyramid_levels
        encoder_features = encoder_features[-self.num_pyramid_levels:]

        # Build lateral features
        if self.use_lateral:
            lateral_features = [
                lateral_conv(feat)
                for lateral_conv, feat in zip(self.lateral_convs, encoder_features)
            ]
        else:
            lateral_features = [
                direct_conv(feat)
                for direct_conv, feat in zip(self.direct_convs, encoder_features)
            ]

        # Build top-down pathway
        if self.use_topdown and hasattr(self, 'fpn_convs'):
            fpn_features = []

            # Start from deepest feature (smallest spatial size)
            prev_feature = lateral_features[-1]
            fpn_features.append(self.fpn_convs[-1](prev_feature))

            # Build pyramid top-down
            for i in range(len(lateral_features) - 2, -1, -1):
                # Upsample previous feature
                upsampled = F.interpolate(
                    prev_feature,
                    size=lateral_features[i].shape[2:],
                    mode='nearest'
                )

                # Merge with lateral connection
                merged = upsampled + lateral_features[i]

                # Apply convolutions
                fpn_feature = self.fpn_convs[i](merged)
                fpn_features.insert(0, fpn_feature)

                prev_feature = merged

            # Use highest resolution feature for segmentation
            final_feature = fpn_features[0]
        else:
            # No top-down pathway - just use highest resolution lateral feature
            final_feature = lateral_features[0]

        # Segmentation head
        output = self.seg_blocks(final_feature)

        # Upsample to input resolution (4x for ResNet-style encoders)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)

        return output


class CustomFPN(nn.Module):
    """
    Custom FPN built from scratch with pretrained ResNet backbone

    Args:
        backbone_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        pretrained: Use ImageNet pretrained weights (default: True)
        pyramid_channels: Number of channels in FPN pyramid (default: 256)
        segmentation_channels: Channels in segmentation head (default: 128)
        num_classes: Number of output classes (default: 1)
        use_lateral: Enable lateral connections (default: True)
        use_topdown: Enable top-down pathway (default: True)
        num_pyramid_levels: Number of pyramid levels (default: 4)
        decoder_depth: Conv blocks per pyramid level (default: 1)
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(
        self,
        backbone_name='resnet50',
        pretrained=True,
        pyramid_channels=256,
        segmentation_channels=128,
        num_classes=1,
        use_lateral=True,
        use_topdown=True,
        num_pyramid_levels=4,
        decoder_depth=1,
        dropout=0.2
    ):
        super().__init__()

        # Encoder (pretrained ResNet)
        self.encoder = ResNetBackbone(backbone_name, pretrained)

        # Decoder (custom FPN)
        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            num_classes=num_classes,
            use_lateral=use_lateral,
            use_topdown=use_topdown,
            num_pyramid_levels=num_pyramid_levels,
            decoder_depth=decoder_depth,
            dropout=dropout
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Extract multi-scale features
        encoder_features = self.encoder(x)

        # FPN decoder
        output = self.decoder(encoder_features)

        # Ensure output matches input size
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output


# ============================================================================
# Factory functions for different ablation configurations
# ============================================================================

def create_fpn_full(backbone_name='resnet50', **kwargs):
    """Full FPN with all components (baseline)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=True,
        use_topdown=True,
        num_pyramid_levels=4,
        decoder_depth=1,
        **kwargs
    )


def create_fpn_no_lateral(backbone_name='resnet50', **kwargs):
    """FPN without lateral connections (tests skip connection importance)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=False,
        use_topdown=True,
        num_pyramid_levels=4,
        decoder_depth=1,
        **kwargs
    )


def create_fpn_no_topdown(backbone_name='resnet50', **kwargs):
    """FPN without top-down pathway (no multi-scale fusion)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=True,
        use_topdown=False,
        num_pyramid_levels=4,
        decoder_depth=1,
        **kwargs
    )


def create_fpn_shallow(backbone_name='resnet50', **kwargs):
    """FPN with fewer pyramid levels (tests multi-scale importance)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=True,
        use_topdown=True,
        num_pyramid_levels=2,  # Only use 2 highest resolution levels
        decoder_depth=1,
        **kwargs
    )


def create_fpn_deep_decoder(backbone_name='resnet50', **kwargs):
    """FPN with deeper decoder (more processing at each level)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=True,
        use_topdown=True,
        num_pyramid_levels=4,
        decoder_depth=2,  # 2 conv blocks per level instead of 1
        **kwargs
    )


def create_fpn_minimal(backbone_name='resnet50', **kwargs):
    """Minimal FPN (no lateral, no top-down, single scale)"""
    return CustomFPN(
        backbone_name=backbone_name,
        use_lateral=False,
        use_topdown=False,
        num_pyramid_levels=1,  # Just the highest resolution
        decoder_depth=0,
        **kwargs
    )


if __name__ == '__main__':
    print("Testing FPN Ablation Configurations\n")
    print("="*70)

    configs = [
        ('Full FPN (Baseline)', create_fpn_full),
        ('No Lateral Connections', create_fpn_no_lateral),
        ('No Top-Down Pathway', create_fpn_no_topdown),
        ('Shallow (2 levels)', create_fpn_shallow),
        ('Deep Decoder (2 blocks)', create_fpn_deep_decoder),
        ('Minimal FPN', create_fpn_minimal),
    ]

    # Test input - batch of 2 RGB images at 256x256
    x = torch.randn(2, 3, 256, 256)

    for name, create_fn in configs:
        model = create_fn(backbone_name='resnet50')
        model.eval()

        with torch.no_grad():
            output = model(x)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        print(f"{name:30s} -> Output: {tuple(output.shape)}, Params: {params:,}")

    print("="*70)
    print("\nAll configurations working correctly!")
