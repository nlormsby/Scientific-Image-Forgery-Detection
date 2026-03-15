import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import itertools

# --- 1. Custom block structures for Unet-style architecture ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x): return F.relu(self.conv(x) + self.shortcut(x))

# --- 2. Pretrained Backbone (variable) + Custom Decoder ---

class PretrainedHybridUNet(nn.Module):
    def __init__(self, n_classes, block_type=DoubleConv, backbone='resnet34'):
        super().__init__()
        
        # Load Pretrained Encoder
        if backbone == 'resnet34':
            base = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            base = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            filters = [64, 64, 128, 256, 512]
            
        # Encoder Layers (Pretrained)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu) # 64
        self.enc1 = base.layer1 # 64
        self.enc2 = base.layer2 # 128
        self.enc3 = base.layer3 # 256
        self.enc4 = base.layer4 # 512

        # Decoder Layers (using custom blocks)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = block_type(filters[3] * 2, filters[3])
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = block_type(filters[2] * 2, filters[2])
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = block_type(filters[1] * 2, filters[1])
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = block_type(filters[0] * 2, filters[0])
        
        self.final_up = nn.ConvTranspose2d(filters[0], 32, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Downward path (Pretrained)
        s0 = self.enc0(x)      # 1/2 size
        s1 = self.enc1(s0)     # 1/4 size
        s2 = self.enc2(s1)     # 1/8 size
        s3 = self.enc3(s2)     # 1/16 size
        s4 = self.enc4(s3)     # 1/32 size

        # Upward path (Custom Blocks + Skip Connections)
        x = self.up4(s4)
        x = torch.cat([x, s3], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, s0], dim=1)
        x = self.dec1(x)
        
        x = self.final_up(x)
        return self.outc(x)

# --- 3. Data loader (needs to be implemented) ---

class MyCustomDataset(Dataset):
    def __init__(self, **kwargs):
        # TODO: Implement local path loading here
        pass
    def __len__(self):
        return 100 # Placeholder
    def __getitem__(self, idx):
        # TODO: Return (Image_Tensor, Mask_Tensor)
        return torch.randn(3, 224, 224), torch.randint(0, 12, (224, 224))

# --- 4. Hyperparam + Multi-Config Training Loop ---

def run_training_loop(n_classes=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define search grid
    search_space = {
        "backbone": ["resnet18", "resnet34"],
        "block": [DoubleConv, ResBlock],
        "optimizer": ["Adam", "SGD"],
        "loss_fn": ["CrossEntropy", "Dice"]
    }
    
    # Generate combinations
    keys, values = zip(*search_space.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. DATA LOADERS
    train_ds = MyCustomDataset() 
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    results = []

    for i, cfg in enumerate(configs):
        print(f"\nExperiment {i+1}/{len(configs)}: {cfg}")
        
        model = PretrainedHybridUNet(n_classes, cfg['block'], cfg['backbone']).to(device)
        
        # Optimizer logic
        if cfg['optimizer'] == "Adam":
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            
        # Loss logic
        criterion = nn.CrossEntropyLoss() # Add Dice logic if needed
        
        # Minimal training loop
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            if batch_idx > 5: break # Short-circuit for testing
            
        results.append({**cfg, "loss": epoch_loss / 6})
        
        # Memory safety
        del model, opt
        torch.cuda.empty_cache()

    return sorted(results, key=lambda x: x['loss'])

# --- 5. Run file ---
if __name__ == "__main__":
    leaderboard = run_training_loop()
    print("\n" + "="*50 + "\nLEADERBOARD\n" + "="*50)
    for entry in leaderboard:
        print(entry)