import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))
import legacy

def load_stylegan_model(model_path, device):
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    return G

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Load Model
    model_path = "checkpoints/ffhq.pkl"
    if not os.path.exists(model_path):
        import glob
        matches = glob.glob("checkpoints/stylegan2_ada/*.pkl")
        if matches: model_path = matches[0]
    
    print(f"Loading model: {model_path}")
    G = load_stylegan_model(model_path, device)
    
    # 2. Generate Random W samples
    print("Generating random W samples...")
    num_samples = 2000
    z = torch.randn(num_samples, G.z_dim).to(device)
    w = G.mapping(z, None)
    w = w[:, 0, :].cpu().numpy()
    
    # 3. Load Attribute Vectors
    vector_dir = "output/vectors"
    target_attrs = ['Eyeglasses', 'Eyeglasses_balanced', 'Male', 'Young', 'Smiling', 'Blond_Hair']
    vectors = {}
    
    for name in target_attrs:
        path = os.path.join(vector_dir, f"vector_{name}.npy")
        if os.path.exists(path):
            v = np.load(path)
            # Normalize and scale for visualization
            v = v / np.linalg.norm(v) * 10  # Scale up more for better visibility
            vectors[name] = v
            
    if not vectors:
        print("No vectors found.")
        return

    # 4. PCA Projection (3D)
    print("Computing PCA (3 Components)...")
    pca = PCA(n_components=3)
    w_pca = pca.fit_transform(w)
    
    w_mean = np.mean(w, axis=0)
    mean_pca = pca.transform(w_mean.reshape(1, -1))[0]
    
    vec_pca = {}
    for name, v in vectors.items():
        end_point = w_mean + v
        end_pca = pca.transform(end_point.reshape(1, -1))[0]
        vec_pca[name] = end_pca - mean_pca

    # 5. Plotting (Dark Theme)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis background for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    # Plot W distribution (faint stars)
    ax.scatter(w_pca[:, 0], w_pca[:, 1], w_pca[:, 2], 
               alpha=0.15, c='white', s=1, label=f'W Space (N={num_samples})')
    
    # Plot Vectors
    colors = {
        'Eyeglasses': '#FF0055',          # Neon Red
        'Eyeglasses_balanced': '#00FF00', # Neon Green
        'Male': '#00CCFF',                # Cyan
        'Young': '#FFAA00',               # Orange
        'Smiling': '#FFFF00',             # Yellow
        'Blond_Hair': '#FF00FF'           # Magenta
    }
    
    origin = mean_pca
    
    for name, v in vec_pca.items():
        color = colors.get(name, 'white')
        label = name.replace('_balanced', ' (Debiased)')
        if 'balanced' in name:
            label += " [N=80]"
        elif name in ['Male', 'Young', 'Smiling', 'Blond_Hair', 'Eyeglasses']:
            label += " [N=200]"
        
        # Draw Arrow
        ax.quiver(origin[0], origin[1], origin[2], 
                  v[0], v[1], v[2], 
                  color=color, arrow_length_ratio=0.1, linewidth=3.5, label=label)
        
        # Add Text Label
        ax.text(origin[0] + v[0]*1.15, origin[1] + v[1]*1.15, origin[2] + v[2]*1.15, 
                label, color=color, fontsize=11, fontweight='bold')

    ax.set_title(f'Latent Space Dynamics (3D PCA)\nBackground Samples: {num_samples}', fontsize=16, color='white', pad=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Set view angle for best perspective
    ax.view_init(elev=20, azim=45)
    
    plt.legend(loc='upper right', facecolor='black', edgecolor='white')
    
    output_dir = "output/analysis"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'latent_space_3d.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
