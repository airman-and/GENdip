import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
            self, 
            latent_dim=128, 
            image_size=64
        ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.dynamic_size = self.image_size
        for _ in range(4):
            self.dynamic_size = math.floor((self.dynamic_size - 4 + 2 * 1) / 2) + 1
        self.flatten_size = 256 * self.dynamic_size * self.dynamic_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        )

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x_enc = self.encoder(x)
        x_enc = torch.flatten(x_enc, start_dim=1)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x_dec = self.decoder_input(z)
        x_dec = x_dec.reshape(-1, 256, self.dynamic_size, self.dynamic_size)
        x_recon = self.decoder(x_dec)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def get_vae_model(
        model_path: str,
        model_latent_dim: int, 
        image_size: int,
        device: str
    ):
    model = VAE(latent_dim=model_latent_dim, image_size=image_size)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'fc_decode.weight' or k == 'fc_decode.bias':
                new_key = k.replace('fc_decode', 'decoder_input')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k == 'fc_decode.weight' or k == 'fc_decode.bias':
                new_key = k.replace('fc_decode', 'decoder_input')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)    
    print(f"[Model] get_vae_model success ({model_path})")

    model = model.to(device)
    model.eval()
    return model

