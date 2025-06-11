import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=15, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Output layer
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skips = skips[::-1]

        # Decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # ConvTranspose2d
            skip = skips[i // 2]
            
           
            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])
            
          
            x = torch.cat((x, skip), dim=1)
            
            
            x = self.decoder[i + 1](x)  # DoubleConv

        return self.output_conv(x)
    
def test_unet():
    model = UNet(in_channels=4, out_channels=15)
    x = torch.randn((1, 4, 240, 240)) 
    output = model(x)
    print(f"Output shape: {output.shape}")  # Dovrebbe essere (1, 15, 160, 160)

#test_unet()