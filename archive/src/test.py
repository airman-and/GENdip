import os
import torch
from torchvision.utils import make_grid, save_image

import config
from core.dataset import get_celeba_loader
from core.model import get_vae_model


def test():
    if not os.path.exists(config.model_path):
        print("[Main] model_path does not exist")
        return
    model = get_vae_model(
        model_path = config.model_path, 
        model_latent_dim = config.model_latent_dim,
        image_size = config.image_size,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    if not os.path.exists(config.celebA_image_path) or not os.path.exists(config.celebA_attr_path):
        print("[Test] celebA_image_path or celebA_attr_path  does not exist")
        return
    celeba_loader = get_celeba_loader(
        celebA_image_path = config.celebA_image_path,
        celebA_attr_path = config.celebA_attr_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=config.shuffle
    )

    if not os.path.exists(config.output_path):
        print("[Test] output_path does not exist")
        return
    
    num_samples = 8
    images = next(iter(celeba_loader))[0][:num_samples].to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        enc = model.encode(images)
        mu, logvar = enc[0], enc[1]
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z).clamp(0.0, 1.0)

    grid = make_grid(torch.cat([images.cpu(), recon.cpu()]), nrow=num_samples)
    out_path = os.path.join(config.output_path, 'test.png')
    save_image(grid, out_path)
    print(f"[Test] Test Success ({out_path})")



if __name__ == '__main__':
    test()

