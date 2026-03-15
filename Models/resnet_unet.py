import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # Pretrained Encoder
        base = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc1, self.enc2, self.enc3, self.enc4 = base.layer1, base.layer2, base.layer3, base.layer4

        # Decoder with Skip Connections
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.Conv2d(32, n_classes, 1),
            nn.Sigmoid() # For Binary Forgery Mask
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.final(d2)

# Placeholder for manual training loop
if __name__ == "__main__":
    model = ResNetUNet()
    print("ResNet-UNet Initialized with Pretrained Weights.")