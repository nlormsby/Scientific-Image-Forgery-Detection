import torch
import torch.nn as nn
import torchvision.models as models

class ViTSegmenter(nn.Module):
    def __init__(self, backbone_name='vit_b_16', n_classes=1):
        super().__init__()
        # Pretrained ViT Backbone
        # Options: vit_b_16, vit_l_16
        weights = models.get_model_weights(backbone_name).DEFAULT
        self.vit = models.get_model(backbone_name, weights=weights)
        
        # ViT typically outputs [Batch, Seq_Len, Hidden_Dim]
        # We transform these tokens back into a 2D grid
        self.hidden_dim = self.vit.hidden_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=16, stride=16), # Match ViT patch size
            nn.Conv2d(64, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through ViT to get patch embeddings
        # (Assuming standard torchvision ViT forward pass)
        b = x.shape[0]
        x = self.vit._process_input(x)
        n = x.shape[1]
        
        # Add class token and position embedding
        cls_token = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)
        
        # Remove class token and reshape to grid
        # For a 224x224 image with patch 16, grid is 14x14
        features = x[:, 1:, :].transpose(1, 2).reshape(b, self.hidden_dim, 14, 14)
        return self.decoder(features)

if __name__ == "__main__":
    model = ViTSegmenter()
    print("ViT Segmenter Initialized.")