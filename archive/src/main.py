import os
import torch
import numpy as np
from torchvision.utils import save_image

import config
from core.dataset import get_celeba_loader
from core.aae_model import Encoder, Decoder


def extract_attribute_vector(encoder, dataloader, attr_name, device, num_samples=100):
    """
    Extract attribute vector by averaging latent codes of images with/without attribute.
    
    Args:
        encoder: Trained encoder model
        dataloader: DataLoader with attribute filtering
        attr_name: Name of the attribute (e.g., "Smiling")
        device: torch device
        num_samples: Number of samples to average
    
    Returns:
        attribute_vector: The difference vector representing the attribute
    """
    encoder.eval()
    
    # Get dataloader for images WITH the attribute
    loader_with_attr = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=config.image_size,
        filter_attr=attr_name,
        filter_value=1,
        shuffle=False,
        max_samples=num_samples
    )
    
    # Get dataloader for images WITHOUT the attribute
    loader_without_attr = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=config.image_size,
        filter_attr=attr_name,
        filter_value=-1,
        shuffle=False,
        max_samples=num_samples
    )
    
    # Encode images with attribute
    z_with = []
    with torch.no_grad():
        for images, _ in loader_with_attr:
            images = images.to(device)
            z = encoder(images)
            z_with.append(z.cpu())
    z_with = torch.cat(z_with, dim=0)
    z_with_mean = z_with.mean(dim=0)
    
    # Encode images without attribute
    z_without = []
    with torch.no_grad():
        for images, _ in loader_without_attr:
            images = images.to(device)
            z = encoder(images)
            z_without.append(z.cpu())
    z_without = torch.cat(z_without, dim=0)
    z_without_mean = z_without.mean(dim=0)
    
    # Compute attribute vector
    attribute_vector = z_with_mean - z_without_mean
    
    print(f"[Attribute] Extracted '{attr_name}' vector from {len(z_with)} positive and {len(z_without)} negative samples")
    
    return attribute_vector.to(device)


def manipulate_image(encoder, decoder, image, attribute_vector, scale=3.0, device='cuda'):
    """
    Manipulate an image by adding an attribute vector to its latent code.
    
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        image: Input image tensor
        attribute_vector: Attribute direction vector in latent space
        scale: Scaling factor for the attribute vector
        device: torch device
    
    Returns:
        manipulated_image: Image with attribute added
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode original image
        z = encoder(image.to(device))
        
        # Add attribute vector
        z_manipulated = z + scale * attribute_vector
        
        # Decode manipulated latent code
        manipulated_image = decoder(z_manipulated)
        
    return manipulated_image


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Main] Device: {device}")
    
    # Check if AAE model exists
    if not os.path.exists(config.aae_model_path):
        print(f"[Main] AAE model not found at {config.aae_model_path}")
        print(f"[Main] Please train the model first by running: python src/train_aae.py")
        return
    
    # Load AAE model
    print(f"[Main] Loading AAE model from {config.aae_model_path}")
    encoder = Encoder(latent_dim=config.model_latent_dim, image_size=config.image_size).to(device)
    decoder = Decoder(latent_dim=config.model_latent_dim, image_size=config.image_size).to(device)
    
    checkpoint = torch.load(config.aae_model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print("[Main] Model loaded successfully")
    
    # Check if dataset exists
    if not os.path.exists(config.celebA_image_path):
        print(f"[Main] Dataset not found at {config.celebA_image_path}")
        return
    
    # Extract attribute vector (e.g., "Smiling")
    print("\n[Main] Extracting 'Smiling' attribute vector...")
    smiling_vector = extract_attribute_vector(
        encoder=encoder,
        dataloader=None,
        attr_name="Smiling",
        device=device,
        num_samples=100
    )
    
    # Load test images (without smiling)
    print("\n[Main] Loading test images...")
    test_loader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=8,
        image_size=config.image_size,
        filter_attr="Smiling",
        filter_value=-1,
        shuffle=True,
        max_samples=8
    )
    
    # Get a batch of test images
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    
    # Manipulate images with different scales
    print("\n[Main] Manipulating images...")
    results = [test_images]  # Original images
    
    for scale in [1.0, 2.0, 3.0]:
        manipulated = manipulate_image(
            encoder=encoder,
            decoder=decoder,
            image=test_images,
            attribute_vector=smiling_vector,
            scale=scale,
            device=device
        )
        results.append(manipulated)
    
    # Save results
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Concatenate all results
    all_results = torch.cat(results, dim=0)
    output_path_file = os.path.join(output_dir, 'latent_control_smiling.png')
    
    save_image(
        all_results,
        output_path_file,
        nrow=8,
        normalize=True
    )
    
    print(f"\n[Main] Results saved to {output_path_file}")
    print("[Main] Rows from top to bottom:")
    print("  - Row 1: Original images (no smiling)")
    print("  - Row 2: Scale 1.0 (slight smile)")
    print("  - Row 3: Scale 2.0 (moderate smile)")
    print("  - Row 4: Scale 3.0 (strong smile)")


if __name__ == '__main__':
    main()
