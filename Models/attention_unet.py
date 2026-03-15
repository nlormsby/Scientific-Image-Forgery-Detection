import torch
import torch.nn as nn
import torchvision.models as models

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
    def __init__(self, backbone_name='vgg16', n_classes=1):
        super().__init__()
        # Fetching VGG features
        weights = models.get_model_weights(backbone_name).DEFAULT
        vgg = models.get_model(backbone_name, weights=weights).features
        
        self.enc1 = vgg[:4]   # 64
        self.enc2 = vgg[5:9]  # 128
        self.enc3 = vgg[10:16] # 256
        self.enc4 = vgg[17:23] # 512

        self.up = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        self.outc = nn.Sequential(nn.Conv2d(256, n_classes, 1), nn.Sigmoid())

    def forward(self, x):
        e1, e2, e3, e4 = self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x))), self.enc4(self.enc3(self.enc2(self.enc1(x))))
        
        g = self.up(e4)
        x_att = self.att(g=g, x=e3)
        d = self.dec(torch.cat([g, x_att], dim=1))
        return nn.functional.interpolate(self.outc(d), size=x.shape[2:], mode='bilinear')

if __name__ == "__main__":
    for bb in ['vgg11', 'vgg16']:
        model = AttentionUNet(backbone_name=bb)
        print(f"Initialized Attention-UNet with {bb}.")