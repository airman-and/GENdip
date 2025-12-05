import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from core.dataset import get_celeba_loader
from core.aae_model import Encoder, Decoder, Discriminator


def train_aae(
    num_epochs=10,
    batch_size=128,
    image_size=128,
    latent_dim=128,
    learning_rate=0.0002,
    beta1=0.5,
    device='cuda'
):
    # Create output directories
    checkpoint_dir = os.path.join(config.project_dir, 'checkpoints')
    sample_dir = os.path.join(config.output_path, 'aae_samples')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Initialize models
    encoder = Encoder(latent_dim=latent_dim, image_size=image_size).to(device)
    decoder = Decoder(latent_dim=latent_dim, image_size=image_size).to(device)
    discriminator = Discriminator(latent_dim=latent_dim).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"[Training] Using {torch.cuda.device_count()} GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        discriminator = nn.DataParallel(discriminator)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        betas=(beta1, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(beta1, 0.999)
    )
    
    # Resume training if checkpoint exists
    start_epoch = 0
    latest_checkpoint = None
    
    # Find latest checkpoint
    # Note: Since we changed the model architecture, we should NOT load old checkpoints
    # unless they are from the new architecture.
    # For now, let's start fresh or check if the user wants to force resume.
    # To be safe, I will disable auto-resume for this run to ensure we train the new model from scratch.
    # If you want to resume later, you can re-enable this logic.
    
    # if os.path.exists(checkpoint_dir):
    #     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    #     if checkpoints:
    #         # Sort by epoch number assuming format 'aae_epoch_X_latent_Y.pth'
    #         try:
    #             checkpoints.sort(key=lambda x: int(x.split('_')[2]))
    #             latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    #         except:
    #             pass

    if latest_checkpoint:
        print(f"[Training] Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"[Training] Resumed from epoch {start_epoch}")
    
    # Data loader
    if not os.path.exists(config.celebA_image_path):
        print(f"[Error] CelebA dataset not found at {config.celebA_image_path}")
        return
    
    dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True
    )
    
    print(f"[Training] Starting AAE training for {num_epochs} epochs")
    print(f"[Training] Device: {device}")
    print(f"[Training] Batch size: {batch_size}")
    print(f"[Training] Latent dim: {latent_dim}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_recon_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (real_images, _) in enumerate(pbar):
                batch_size_actual = real_images.size(0)
                real_images = real_images.to(device)
                
                # Adversarial ground truths
                valid = torch.ones(batch_size_actual, 1).to(device)
                fake = torch.zeros(batch_size_actual, 1).to(device)
                
                # -----------------
                #  Train Discriminator
                # -----------------
                optimizer_D.zero_grad()
                
                # Sample noise as discriminator ground truth
                z_real = torch.randn(batch_size_actual, latent_dim).to(device)
                
                # Encode images
                z_fake = encoder(real_images)
                
                # Discriminator loss
                real_loss = adversarial_loss(discriminator(z_real), valid)
                fake_loss = adversarial_loss(discriminator(z_fake.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                
                d_loss.backward()
                optimizer_D.step()
                
                # -----------------
                #  Train Generator (Encoder + Decoder)
                # -----------------
                optimizer_G.zero_grad()
                
                # Encode images
                z = encoder(real_images)
                
                # Reconstruct images
                recon_images = decoder(z)
                
                # Generator loss
                recon_loss = reconstruction_loss(recon_images, real_images)
                adv_loss = adversarial_loss(discriminator(z), valid)
                
                # Total generator loss
                g_loss = recon_loss + 0.1 * adv_loss
                
                g_loss.backward()
                optimizer_G.step()
                
                # Update progress bar
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_recon_loss += recon_loss.item()
                
                pbar.set_postfix({
                    'G_loss': f'{g_loss.item():.4f}',
                    'D_loss': f'{d_loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}'
                })
                
                # Save sample images
                if i % 200 == 0:
                    with torch.no_grad():
                        sample_size = min(16, batch_size_actual)
                        comparison = torch.cat([
                            real_images[:sample_size],
                            recon_images[:sample_size]
                        ])
                        save_image(
                            comparison,
                            os.path.join(sample_dir, f'epoch_{epoch+1}_batch_{i}.png'),
                            nrow=sample_size,
                            normalize=True
                        )
        
        # Print epoch statistics
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"G_loss: {avg_g_loss:.4f}, "
              f"D_loss: {avg_d_loss:.4f}, "
              f"Recon_loss: {avg_recon_loss:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'aae_epoch_{epoch+1}_latent_{latent_dim}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict(),
                'decoder_state_dict': decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict(),
                'discriminator_state_dict': discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f"[Checkpoint] Saved to {checkpoint_path}")
    
    print("[Training] AAE training completed!")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_aae(
        num_epochs=30,
        batch_size=768,
        image_size=128,
        latent_dim=128,
        learning_rate=0.0002,
        device=device
    )
