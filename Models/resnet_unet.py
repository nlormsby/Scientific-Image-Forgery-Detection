import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, backbone_name='resnet34', n_classes=1):
        super().__init__()
        
        # 1. Fetch Pretrained Backbone
        # Options: resnet18, resnet34, resnet50
        weights = models.get_model_weights(backbone_name).DEFAULT
        base = models.get_model(backbone_name, weights=weights)
        
        # 2. Slice Encoder
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc1 = base.layer1 # 64 ch
        self.enc2 = base.layer2 # 128 ch
        self.enc3 = base.layer3 # 256 ch
        self.enc4 = base.layer4 # 512 ch (assuming ResNet34/18)

        # 3. Flexible Decoder (Adjusting for ResNet depth)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
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

if __name__ == "__main__":
    for bb in ['resnet18', 'resnet34']:
        model = ResNetUNet(backbone_name=bb)
        print(f"Successfully initialized ResNet-UNet with {bb} backbone.")