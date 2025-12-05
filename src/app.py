import streamlit as st
import os
import sys
import torch
import numpy as np
import pickle
import glob

# Add stylegan2_ada_pytorch to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dnnlib
import legacy

# Page config
st.set_page_config(
    page_title="StyleGAN2-ADA Latent Control",
    layout="wide"
)

@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find model
    model_path = "checkpoints/ffhq.pkl"
    if not os.path.exists(model_path):
        matches = glob.glob("checkpoints/stylegan2_ada/*.pkl")
        if matches: model_path = matches[0]
        else:
             matches = glob.glob("output/stylegan2_ada_training/**/network-snapshot-*.pkl", recursive=True)
             if matches: model_path = max(matches, key=os.path.getmtime)
    
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    return G, device

@st.cache_data
def load_vectors():
    vectors = {}
    vector_dir = "output/vectors"
    if not os.path.exists(vector_dir):
        return vectors
        
    for f in glob.glob(os.path.join(vector_dir, "*.npy")):
        name = os.path.basename(f).replace("vector_", "").replace(".npy", "")
        vectors[name] = np.load(f)
    return vectors

def main():
    st.title("StyleGAN2-ADA Latent Control Demo")
    
    # Load resources
    with st.spinner("Loading Model..."):
        G, device = load_model()
    
    vectors = load_vectors()
    if not vectors:
        st.error("No attribute vectors found. Please run 'python src/save_vectors.py' first.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Seed control
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    if st.sidebar.button("Randomize"):
        seed = np.random.randint(0, 100000)
        st.rerun()
        
    # Attribute sliders
    st.sidebar.subheader("Attributes")
    coeffs = {}
    for name in vectors.keys():
        coeffs[name] = st.sidebar.slider(f"{name}", -3.0, 3.0, 0.0, 0.1)

    # Session state for W
    if 'w' not in st.session_state or st.session_state.seed != seed:
        st.session_state.seed = seed
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        st.session_state.w = G.mapping(z, None)
    
    w = st.session_state.w
    
    # Apply edits
    w_edited = w.clone()
    for name, coeff in coeffs.items():
        if coeff != 0:
            v = torch.from_numpy(vectors[name]).to(device)
            v = v / torch.norm(v)
            w_edited = w_edited + (v * coeff)
            
    # Synthesis
    with torch.no_grad():
        img = G.synthesis(w_edited, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        # Resize for faster display (optional, but helps responsiveness)
        # img_small = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2).float(), size=(512, 512), mode='area').permute(0, 2, 3, 1).to(torch.uint8)
        # img_np = img_small[0].cpu().numpy()
        img_np = img[0].cpu().numpy() # Keep original quality for now as 60fps is fast enough

    # Display
    col1, col2 = st.columns([3, 1]) # Adjust layout
    with col1:
        st.image(img_np, caption=f"Result (Seed: {seed})", use_container_width=True)
    
    with col2:
        st.write("### Applied Attributes")
        for name, coeff in coeffs.items():
            if coeff != 0:
                st.write(f"- **{name}:** {coeff}")

if __name__ == "__main__":
    main()
