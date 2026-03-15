import torch
import torch.nn as nn
import torchvision.models as models

class MantraStreamNet(nn.Module):
    def __init__(self, backbone_name='resnet18', n_classes=1):
        super().__init__()
        # Stream A: RGB Backbone (Pretrained)
        weights = models.get_model_weights(backbone_name).DEFAULT
        self.rgb_stream = nn.Sequential(*list(models.get_model(backbone_name, weights=weights).children())[:-2])
        
        # Stream B: Noise Stream (SRM filter inspired)
        # Learns to find inconsistencies in image grain/noise
        self.noise_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )

        # Fusion and Localization
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 64, 256, 3, padding=1), # Adjust 512 based on backbone
            nn.ReLU(),
            nn.Conv2d(256, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_rgb = self.rgb_stream(x)
        f_noise = self.noise_stream(x)
        
        # Upsample noise to match RGB feature size
        f_noise = nn.functional.interpolate(f_noise, size=f_rgb.shape[2:])
        combined = torch.cat([f_rgb, f_noise], dim=1)
        
        # Output probability map
        out = self.fusion(combined)
        return nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear')

if __name__ == "__main__":
    model = MantraStreamNet(backbone_name='resnet18')
    print("MantraStreamNet Initialized.")