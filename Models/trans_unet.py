import torch
import torch.nn as nn

class TransUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # Pretrained ResNet Encoder
        import torchvision.models as models
        resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Features
        
        # Transformer Bottleneck
        self.vit_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.vit_layer, num_layers=4)

        # Decoder
        self.up = nn.ConvTranspose2d(512, 256, 2, 2)
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x) # [B, 512, H/32, W/32]
        
        # Flatten for Transformer
        b, c, h, w = features.shape
        x_flat = features.flatten(2).permute(2, 0, 1) # [HW, B, C]
        x_trans = self.transformer(x_flat)
        x_res = x_trans.permute(1, 2, 0).reshape(b, c, h, w)
        
        # Upsample
        up = self.up(x_res)
        return nn.functional.interpolate(self.out(up), size=x.shape[2:])

if __name__ == "__main__":
    model = TransUNet()
    print("TransUNet (Hybrid) Initialized.")