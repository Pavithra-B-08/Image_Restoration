import torch
import torch.nn as nn

# Simple U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.dec1 = block(512, 256)
        self.dec2 = block(256, 128)
        self.dec3 = block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d1 = self.upconv1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.final(d3)
        return out
    
# Add this code right after your device setup and before dataset creation
print("Testing UNet model initialization...")
try:
    test_model = UNet()
    print(f"UNet model initialized successfully: {type(test_model)}")
    print(f"UNet model architecture: {test_model}")
except Exception as e:
    print(f"Error initializing UNet model: {e}")
    import traceback
    traceback.print_exc()
