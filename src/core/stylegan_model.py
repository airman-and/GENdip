import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualizedLinear(nn.Module):
    """Equalized Learning Rate Linear Layer"""
    def __init__(self, in_dim, out_dim, lr_mul=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
        self.lr_mul = lr_mul
        
    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, self.bias * self.lr_mul if self.bias is not None else None)


class EqualizedConv2d(nn.Module):
    """Equalized Learning Rate Conv2d Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul
        
    def forward(self, x):
        return F.conv2d(x, self.weight * self.lr_mul, self.bias * self.lr_mul if self.bias is not None else None, 
                       stride=self.stride, padding=self.padding)


class NoiseInjection(nn.Module):
    """Noise Injection Module"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise


class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = EqualizedLinear(style_dim, channels, lr_mul=1.0)
        self.style_bias = EqualizedLinear(style_dim, channels, lr_mul=1.0)
        
    def forward(self, x, style):
        # Normalize
        x = self.norm(x)
        
        # Apply style
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        
        return style_scale * x + style_bias


class StyleBlock(nn.Module):
    """StyleGAN Synthesis Block"""
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = EqualizedConv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.noise = NoiseInjection(out_channels)
        self.adain = AdaIN(style_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, style, noise=None):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.activation(x)
        x = self.adain(x, style)
        return x


class MappingNetwork(nn.Module):
    """Mapping Network: Z -> W"""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=0.01))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.mapping(z)


class SynthesisNetwork(nn.Module):
    """StyleGAN Synthesis Network"""
    def __init__(self, w_dim=512, image_size=128, channels=512):
        super().__init__()
        self.w_dim = w_dim
        self.image_size = image_size
        self.num_layers = int(np.log2(image_size)) - 1  # 128 -> 7 layers (4x4 to 128x128)
        
        # Initial constant
        self.const = nn.Parameter(torch.ones(1, channels, 4, 4))
        
        # Style blocks
        self.style_blocks = nn.ModuleList()
        
        # 4x4
        self.style_blocks.append(StyleBlock(channels, channels, w_dim, upsample=False))
        
        # 8x8 to 128x128
        in_ch = channels
        for i in range(1, self.num_layers):
            if i == self.num_layers - 1:  # Last layer
                out_ch = 32
            elif i >= self.num_layers - 3:  # Last 3 layers
                out_ch = in_ch // 2
            else:
                out_ch = in_ch
            self.style_blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=True))
            in_ch = out_ch
        
        # To RGB
        self.to_rgb = EqualizedConv2d(in_ch, 3, 1, bias=True)
        
    def forward(self, w, noise=None):
        """
        Args:
            w: [batch, num_layers, w_dim] or [batch, w_dim] (broadcasted)
            noise: List of noise tensors for each layer
        """
        batch_size = w.shape[0] if len(w.shape) == 2 else w.shape[0]
        
        # Handle w shape: if [batch, w_dim], broadcast to [batch, num_layers, w_dim]
        if len(w.shape) == 2:
            w = w.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Start with constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # First style block (no upsample)
        x = self.style_blocks[0](x, w[:, 0], noise[0] if noise else None)
        
        # Remaining style blocks
        for i in range(1, self.num_layers):
            x = self.style_blocks[i](x, w[:, i], noise[i] if noise else None)
        
        # To RGB
        x = self.to_rgb(x)
        x = torch.tanh(x)  # [-1, 1] range
        
        return x


class StyleGANGenerator(nn.Module):
    """StyleGAN Generator"""
    def __init__(self, z_dim=512, w_dim=512, image_size=128, num_mapping_layers=8, channels=512):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.image_size = image_size
        
        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping_layers)
        self.synthesis = SynthesisNetwork(w_dim, image_size, channels)
        
    def forward(self, z, noise=None, return_w=False, truncation=1.0, truncation_mean=None):
        """
        Args:
            z: Latent code [batch, z_dim]
            noise: List of noise tensors (optional)
            return_w: Whether to return w codes
            truncation: Truncation trick value
            truncation_mean: Mean w for truncation
        """
        # Map to W space
        w = self.mapping(z)
        
        # Truncation trick
        if truncation < 1.0 and truncation_mean is not None:
            w = truncation_mean + truncation * (w - truncation_mean)
        
        # Generate noise if not provided
        if noise is None:
            noise = self.make_noise(batch_size=z.shape[0])
        
        # Generate image
        img = self.synthesis(w, noise)
        
        if return_w:
            return img, w
        return img
    
    def make_noise(self, batch_size=1):
        """Generate random noise for each layer"""
        num_layers = self.synthesis.num_layers
        noise = []
        for i in range(num_layers):
            # Calculate resolution for this layer
            res = 4 * (2 ** i)
            noise.append(torch.randn(batch_size, 1, res, res, device=next(self.parameters()).device))
        return noise


class StyleGANDiscriminator(nn.Module):
    """StyleGAN Discriminator"""
    def __init__(self, image_size=128, channels=32):
        super().__init__()
        self.image_size = image_size
        num_layers = int(np.log2(image_size)) - 1
        
        # From RGB
        self.from_rgb = EqualizedConv2d(3, channels, 1, bias=True)
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        
        in_ch = channels
        for i in range(num_layers - 1, 0, -1):  # From high res to low res
            out_ch = min(in_ch * 2, 512)
            self.blocks.append(self._make_block(in_ch, out_ch, downsample=True))
            in_ch = out_ch
        
        # Final block (no downsample)
        self.blocks.append(self._make_block(in_ch, 512, downsample=False))
        
        # Final layers
        self.final_conv = EqualizedConv2d(512, 512, 4, bias=True)
        self.final_linear = EqualizedLinear(512, 1, lr_mul=1.0)
        
    def _make_block(self, in_channels, out_channels, downsample=True):
        layers = []
        if downsample:
            layers.append(nn.AvgPool2d(2))
        layers.append(EqualizedConv2d(in_channels, out_channels, 3, padding=1, bias=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(EqualizedConv2d(out_channels, out_channels, 3, padding=1, bias=True))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.from_rgb(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        
        return x

