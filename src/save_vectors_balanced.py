import os
import sys
import torch
import numpy as np
import pandas as pd
import glob
from torch.utils.data import DataLoader
from torchvision import transforms

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.dataset import CelebADataset
from projector import project
import legacy

def load_stylegan_model(model_path, device):
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    return G

def get_balanced_ids(df, target_attr, confounders, num_samples=200):
    """
    Selects balanced samples for Positive and Negative sets of target_attr,
    balancing across all combinations of confounders.
    """
    print(f"Balancing '{target_attr}' against {confounders}...")
    
    def sample_balanced_group(target_val):
        selected_ids = []
        # Generate all binary combinations of confounders
        import itertools
        combinations = list(itertools.product([-1, 1], repeat=len(confounders)))
        
        samples_per_combo = num_samples // len(combinations)
        print(f"  Target {target_val}: Aiming for {samples_per_combo} samples per subgroup ({len(combinations)} groups)")
        
        for combo in combinations:
            # Build query mask
            mask = (df[target_attr] == target_val)
            for i, conf_name in enumerate(confounders):
                mask &= (df[conf_name] == combo[i])
            
            candidates = df[mask]['image_id'].tolist()
            
            if len(candidates) < samples_per_combo:
                print(f"    Warning: Not enough samples for combo {combo}. Found {len(candidates)}, needed {samples_per_combo}.")
                selected_ids.extend(candidates)
            else:
                selected_ids.extend(candidates[:samples_per_combo])
                
        return selected_ids

    pos_ids = sample_balanced_group(1)
    neg_ids = sample_balanced_group(-1)
    
    print(f"  Selected {len(pos_ids)} Positive samples, {len(neg_ids)} Negative samples.")
    return pos_ids, neg_ids

def extract_vector(G, image_ids, device):
    # Create Dataset and Loader
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((G.img_resolution, G.img_resolution)),
        transforms.ToTensor(),
    ])
    
    dataset = CelebADataset(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        transform=transform,
        image_ids=image_ids
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    ws = []
    for i, (images, _) in enumerate(loader):
        img = images[0].to(device)
        img_255 = (img * 255.0).clamp(0, 255).byte()
        w = project(G, img_255, num_steps=50, device=device, verbose=False)
        if w is not None:
            ws.append(w.mean(dim=1, keepdim=True).detach().cpu())
        
        if i % 10 == 0: print(f"    {i}/{len(image_ids)}", end='\r')
            
    if not ws: return None
    return torch.cat(ws, dim=0).mean(dim=0).squeeze(0)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model_path = "checkpoints/ffhq.pkl"
    if not os.path.exists(model_path):
        matches = glob.glob("checkpoints/stylegan2_ada/*.pkl")
        if matches: model_path = matches[0]
    
    print(f"Loading model: {model_path}")
    G = load_stylegan_model(model_path, device)
    
    # Load Attributes
    csv_path = 'dataset/celebA/list_attr_celeba.csv'
    df = pd.read_csv(csv_path)
    
    # 1. Eyeglasses (Balanced against Male, Young)
    target = 'Eyeglasses'
    confounders = ['Male', 'Young']
    
    pos_ids, neg_ids = get_balanced_ids(df, target, confounders, num_samples=80)
    
    print(f"\nProjecting Positive Samples for {target}...")
    w_pos = extract_vector(G, pos_ids, device)
    
    print(f"\nProjecting Negative Samples for {target}...")
    w_neg = extract_vector(G, neg_ids, device)
    
    if w_pos is not None and w_neg is not None:
        attr_vector = (w_pos - w_neg).numpy()
        output_dir = "output/vectors"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'vector_{target}_balanced.npy')
        np.save(save_path, attr_vector)
        print(f"\n[Saved] {save_path}")
    else:
        print("\n[Error] Extraction failed.")

if __name__ == "__main__":
    main()
