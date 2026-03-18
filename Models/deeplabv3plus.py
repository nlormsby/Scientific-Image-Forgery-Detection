import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolution branches with different rates
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Image-level pooling branch (global context)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.conv1x1(x)
        feat2 = self.atrous_conv1(x)
        feat3 = self.atrous_conv2(x)
        feat4 = self.atrous_conv3(x)

        # Image pooling and upsample
        feat5 = self.image_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)

        x = self.project(x)

        return x


class Decoder(nn.Module):
    def __init__(self, low_level_channels, decoder_channels=256, num_classes=1):
        super(Decoder, self).__init__()

        # Reduce low-level feature channels
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder convolutions
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, x, low_level_feat):
        # Reduce low-level feature channels
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample ASPP output 4x to match low-level features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate with low-level features (skip connection)
        x = torch.cat([x, low_level_feat], dim=1)

        # Decoder convolutions
        x = self.decoder_conv(x)

        # Classifier
        x = self.classifier(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_class=1, backbone='resnet34', pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        # Ablation flags
        self.use_aspp = True
        self.use_skip = True
        self.use_image_pool = True

        # Load ResNet backbone
        if backbone == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = torchvision.models.resnet34(weights=weights)
            self.low_level_channels = 64
            self.high_level_channels = 512
        elif backbone == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
            self.low_level_channels = 256
            self.high_level_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Apply atrous convolutions
        self._make_layer_atrous(self.layer4, dilation=2)

        # ASPP module
        self.aspp = ASPP(
            in_channels=self.high_level_channels,
            out_channels=256,
            atrous_rates=[6, 12, 18]
        )

        # Decoder with skip connections
        self.decoder = Decoder(
            low_level_channels=self.low_level_channels,
            decoder_channels=256,
            num_classes=n_class
        )

    def _make_layer_atrous(self, layer, dilation):
        """Apply atrous convolutions to a ResNet layer"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.shape[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)

        # ASPP module
        if self.use_aspp:
            x = self.aspp(high_level_feat)
        else:
            # Ablation: no ASPP, just 1x1 conv
            x = nn.Conv2d(self.high_level_channels, 256, 1).to(high_level_feat.device)(high_level_feat)

        # Decoder with skip connections
        if self.use_skip:
            x = self.decoder(x, low_level_feat)
        else:
            # Ablation: no skip connection, just upsample directly
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
            x = nn.Conv2d(256, 1, 1).to(x.device)(x)

        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


class DeepLabV3Plus_NoASPP(DeepLabV3Plus):
    def __init__(self, n_class=1, backbone='resnet34', pretrained=True):
        super(DeepLabV3Plus_NoASPP, self).__init__(n_class, backbone, pretrained)
        self.use_aspp = False

        # Replace ASPP with simple 1x1 conv
        self.simple_conv = nn.Sequential(
            nn.Conv2d(self.high_level_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)

        # No ASPP - just 1x1 conv
        x = self.simple_conv(high_level_feat)

        # Decoder
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


class DeepLabV3Plus_NoSkip(DeepLabV3Plus):
    def __init__(self, n_class=1, backbone='resnet34', pretrained=True):
        super(DeepLabV3Plus_NoSkip, self).__init__(n_class, backbone, pretrained)
        self.use_skip = False

        # Simpler decoder without skip connections
        self.simple_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, n_class, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)

        # ASPP
        x = self.aspp(high_level_feat)

        # Simple decoder (no skip)
        x = self.simple_decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


# Factory function for easy model creation
def get_deeplabv3plus(variant='default', backbone='resnet34', pretrained=True, n_class=1):
    if variant == 'default':
        return DeepLabV3Plus(n_class=n_class, backbone=backbone, pretrained=pretrained)
    elif variant == 'no_aspp':
        return DeepLabV3Plus_NoASPP(n_class=n_class, backbone=backbone, pretrained=pretrained)
    elif variant == 'no_skip':
        return DeepLabV3Plus_NoSkip(n_class=n_class, backbone=backbone, pretrained=pretrained)
    elif variant == 'minimal':
        model = DeepLabV3Plus_NoSkip(n_class=n_class, backbone=backbone, pretrained=pretrained)
        model.use_aspp = False
        return model
    else:
        raise ValueError(f"Unknown variant: {variant}")
