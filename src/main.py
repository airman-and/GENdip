import os
import torch

import config
from core.dataset import get_celeba_loader
from core.model import get_vae_model


def main():
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
        print("[Main] celebA_image_path or   does not exist")
        return
    celeba_loader = get_celeba_loader(
        celebA_image_path = config.celebA_image_path,
        celebA_attr_path = config.celebA_attr_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=config.shuffle
    )


if __name__ == '__main__':
    main()

