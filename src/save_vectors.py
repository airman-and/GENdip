import os
import sys
import torch
import numpy as np
import pickle
import glob

# Add stylegan2_ada_pytorch to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.dataset import get_celeba_loader
from projector import project
import dnnlib
import legacy

def load_stylegan_model(model_path, device):
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    return G

def extract_and_save_vector(G, attr_name, output_dir, device, num_samples=50):
    print(f"\n[Attribute] Extracting '{attr_name}'...")
    
    # Positive samples
    loader_pos = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr=attr_name,
        filter_value=1,
        shuffle=False,
        max_samples=num_samples
    )
    
    # Negative samples
    loader_neg = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr=attr_name,
        filter_value=-1,
        shuffle=False,
        max_samples=num_samples
    )
    
    w_pos = []
    print("  Projecting positive samples...")
    for i, (images, _) in enumerate(loader_pos):
        if i >= num_samples: break
        img = images[0].to(device)
        img_255 = (img * 255.0).clamp(0, 255).byte()
        w = project(G, img_255, num_steps=200, device=device, verbose=False) # Reduced steps for speed
        if w is not None:
            w_pos.append(w.mean(dim=1, keepdim=True).detach().cpu())
            if i % 10 == 0: print(f"    {i}/{num_samples}", end='\r')
            
    w_neg = []
    print("\n  Projecting negative samples...")
    for i, (images, _) in enumerate(loader_neg):
        if i >= num_samples: break
        img = images[0].to(device)
        img_255 = (img * 255.0).clamp(0, 255).byte()
        w = project(G, img_255, num_steps=200, device=device, verbose=False)
        if w is not None:
            w_neg.append(w.mean(dim=1, keepdim=True).detach().cpu())
            if i % 10 == 0: print(f"    {i}/{num_samples}", end='\r')

    if not w_pos or not w_neg:
        print(f"\n[Error] Failed to extract vector for {attr_name}")
        return

    w_pos_mean = torch.cat(w_pos, dim=0).mean(dim=0).squeeze(0)
    w_neg_mean = torch.cat(w_neg, dim=0).mean(dim=0).squeeze(0)
    attr_vector = (w_pos_mean - w_neg_mean).numpy()
    
    save_path = os.path.join(output_dir, f'vector_{attr_name}.npy')
    np.save(save_path, attr_vector)
    print(f"\n[Saved] {save_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Find model
    model_path = "checkpoints/ffhq.pkl"
    if not os.path.exists(model_path):
        # Fallback search
        matches = glob.glob("checkpoints/stylegan2_ada/*.pkl")
        if matches: model_path = matches[0]
        else:
             # Try network snapshot
             matches = glob.glob("output/stylegan2_ada_training/**/network-snapshot-*.pkl", recursive=True)
             if matches: model_path = max(matches, key=os.path.getmtime)
    
    print(f"Loading model: {model_path}")
    G = load_stylegan_model(model_path, device)
    
    output_dir = "output/vectors"
    os.makedirs(output_dir, exist_ok=True)
    
    # Full list of 40 CelebA attributes
    attributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 
        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
        'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
        'Wearing_Necktie', 'Young'
    ]
    
    print(f"Starting extraction for {len(attributes)} attributes with 2000 samples each...")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    # Split attributes into shards
    shard_size = len(attributes) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else len(attributes)
    
    my_attributes = attributes[start_idx:end_idx]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(my_attributes)} attributes: {my_attributes}")
    
    for attr in my_attributes:
        save_path = os.path.join(output_dir, f'vector_{attr}.npy')
        # Overwrite existing vectors to ensure high quality
        extract_and_save_vector(G, attr, output_dir, device, num_samples=200)

if __name__ == "__main__":
    main()
