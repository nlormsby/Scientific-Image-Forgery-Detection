import torch
import torch.nn as nn
import torchvision.models as models

class TransUNet(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', n_classes=1):
        super().__init__()
        
        # Load Pretrained CNN Backbone
        weights = models.get_model_weights(backbone_name).DEFAULT
        self.backbone = models.get_model(backbone_name, weights=weights).features
        
        # Determine the number of output channels from the backbone automatically
        dummy_in = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_in)
        backbone_out_ch = dummy_out.shape[1]

        # Transformer Bottleneck
        self.vit_layer = nn.TransformerEncoderLayer(d_model=backbone_out_ch, nhead=8)
        self.transformer = nn.TransformerEncoder(self.vit_layer, num_layers=2)

        self.up = nn.ConvTranspose2d(backbone_out_ch, 64, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x) # [B, C, H, W]
        b, c, h, w = feat.shape
        
        # Transformer likes (Sequence, Batch, Channels)
        feat_flat = feat.flatten(2).permute(2, 0, 1)
        feat_trans = self.transformer(feat_flat)
        feat_reshaped = feat_trans.permute(1, 2, 0).reshape(b, c, h, w)
        
        return nn.functional.interpolate(self.final(self.up(feat_reshaped)), size=x.shape[2:])

if __name__ == "__main__":
    for bb in ['efficientnet_b0', 'mobilenet_v3_small']:
        model = TransUNet(backbone_name=bb)
        print(f"Initialized Hybrid-TransUNet with {bb}.")