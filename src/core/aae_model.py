import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, image_size=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial Conv
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # ResNet Blocks (Downsampling)
        self.layer1 = ResNetBlock(64, 128, stride=2)  # 32x32
        self.layer2 = ResNetBlock(128, 256, stride=2) # 16x16
        self.layer3 = ResNetBlock(256, 512, stride=2) # 8x8
        self.layer4 = ResNetBlock(512, 512, stride=2) # 4x4
        
        self.flatten_size = 512 * 4 * 4
        self.fc_z = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        z = self.fc_z(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, image_size=128):
        super(Decoder, self).__init__()
        self.flatten_size = 512 * 4 * 4
        self.fc_input = nn.Linear(latent_dim, self.flatten_size)
        
        # ResNet Blocks (Upsampling)
        # We use Upsample + ResNetBlock(stride=1)
        self.layer1 = self._make_layer(512, 512) # 8x8
        self.layer2 = self._make_layer(512, 256) # 16x16
        self.layer3 = self._make_layer(256, 128) # 32x32
        self.layer4 = self._make_layer(128, 64)  # 64x64
        self.layer5 = self._make_layer(64, 64)   # 128x128
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResNetBlock(in_channels, out_channels, stride=1)
        )

    def forward(self, z):
        x = self.fc_input(z)
        x = x.view(-1, 512, 4, 4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
