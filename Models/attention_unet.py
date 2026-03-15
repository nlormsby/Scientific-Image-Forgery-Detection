import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # Using a simple pretrained VGG-style backbone for simplicity in attention mapping
        import torchvision.models as models
        vgg = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features
        
        self.enc1 = vgg[:4]   # 64
        self.enc2 = vgg[5:9]  # 128
        self.enc3 = vgg[10:16] # 256
        self.enc4 = vgg[17:23] # 512

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = nn.Conv2d(512, 256, 3, padding=1)
        
        self.final = nn.Sequential(nn.Conv2d(256, n_classes, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        g3 = self.up3(e4)
        x3 = self.att3(g=g3, x=e3)
        d3 = self.dec3(torch.cat([g3, x3], dim=1))
        return nn.functional.interpolate(self.final(d3), size=x.shape[2:])

if __name__ == "__main__":
    model = AttentionUNet()
    print("Attention-UNet Initialized.")