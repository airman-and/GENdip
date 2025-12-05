import gradio as gr
import os
import sys
import torch
import numpy as np
import glob
import pickle
import PIL.Image

# Add stylegan2_ada_pytorch to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stylegan2_ada_pytorch'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dnnlib
import legacy
from projector import project
from face_alignment import align_face

# Global variables
G = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vectors = {}
current_w = None # Can be from random Z or projected image
base_w = None    # To store the original W before editing

def load_resources():
    global G, vectors, current_w, base_w
    
    # Load Model
    if G is None:
        print("Loading model...")
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
    
    # Load Vectors
    print("Loading vectors...")
    vector_dir = "output/vectors"
    if os.path.exists(vector_dir):
        for f in glob.glob(os.path.join(vector_dir, "*.npy")):
            name = os.path.basename(f).replace("vector_", "").replace(".npy", "")
            vectors[name] = torch.from_numpy(np.load(f)).to(device)
            # Normalize
            vectors[name] = vectors[name] / torch.norm(vectors[name])
            
    # Init seed if not set
    if current_w is None:
        update_seed(42)

def update_seed(seed):
    global G, current_w, base_w
    if G is None: load_resources()
    
    z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, G.z_dim)).to(device)
    base_w = G.mapping(z, None)
    current_w = base_w.clone()
    return generate_image()

def run_projection(image):
    global G, current_w, base_w
    if G is None: load_resources()
    
    if image is None:
        return None
        
    print("Starting projection...")
    
    # Align Face
    aligned_pil = align_face(image, output_size=G.img_resolution)
    
    if aligned_pil is None:
        # Fallback to simple resize if alignment fails
        print("Alignment failed, falling back to simple resize.")
        target_pil = PIL.Image.fromarray(image).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    else:
        target_pil = aligned_pil
        
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_tensor = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    
    # Run projection
    # Using fewer steps for demo speed (e.g. 500)
    projected_w_steps = project(
        G,
        target=target_tensor,
        num_steps=500,
        device=device,
        verbose=True
    )
    
    # Update global W
    base_w = projected_w_steps[-1].unsqueeze(0) # [1, L, C]
    current_w = base_w.clone()
    
    return generate_image()

def generate_image(**kwargs):
    global G, current_w, base_w, vectors
    
    if G is None: load_resources()
    if base_w is None: update_seed(42)
        
    # Start from base W
    w_edited = base_w.clone()
    
    # Apply edits
    for name, val in kwargs.items():
        if name in vectors and val != 0:
            w_edited += vectors[name] * val
            
    # Synthesis
    with torch.no_grad():
        img = G.synthesis(w_edited, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        
    return img_np

def reset_sliders():
    # Return 0 for all sliders
    return [0.0] * len(vectors)

# Initialize
load_resources()

# Create Interface
with gr.Blocks() as demo:
    gr.Markdown("# StyleGAN2-ADA Latent Control Demo")
    gr.Markdown("Upload your photo or use a random seed to generate a face. Then use the sliders to edit attributes.")
    
    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=1):
            with gr.Tab("Random Generation"):
                seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                btn_random = gr.Button("🎲 Randomize", variant="primary")
            
            with gr.Tab("Image Upload (Inversion)"):
                image_input = gr.Image(label="Upload Face", sources=["upload", "webcam"])
                btn_project = gr.Button("🚀 Invert & Edit", variant="primary")
                gr.Markdown("*Note: Inversion takes about 1-2 minutes.*")
            
            gr.Markdown("### Attribute Controls")
            
            # Dynamic Sliders
            sliders = []
            
            # 1. Main Attributes (Top 5)
            main_attrs = ['Smiling', 'Male', 'Young', 'Eyeglasses', 'Blond_Hair']
            
            gr.Markdown("### ✨ Key Attributes")
            for name in main_attrs:
                if name in vectors:
                    s = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label=name)
                    sliders.append(s)
            
            # 2. Other Attributes (Accordion)
            other_attrs = [k for k in sorted(vectors.keys()) if k not in main_attrs]
            
            with gr.Accordion("🎨 More Attributes", open=False):
                for name in other_attrs:
                    s = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label=name)
                    sliders.append(s)
                
            btn_reset = gr.Button("🔄 Reset Attributes")

        # Right Column: Output
        with gr.Column(scale=2):
            image_output = gr.Image(label="Result", interactive=False)

    # --- Event Handling ---
    
    # Wrapper for generation
    def generate_wrapper(*args):
        # Reconstruct the order: main_attrs + other_attrs
        all_keys = []
        for name in main_attrs:
            if name in vectors: all_keys.append(name)
        all_keys.extend(other_attrs)
        
        kwargs = {}
        for i, name in enumerate(all_keys):
            kwargs[name] = args[i]
        return generate_image(**kwargs)

    # 1. Random Seed
    btn_random.click(
        fn=lambda: np.random.randint(0, 100000), 
        outputs=seed_input
    ).then(
        fn=update_seed,
        inputs=seed_input,
        outputs=image_output
    ).then(
        fn=lambda: [0.0]*len(sliders), # Reset sliders on new seed
        outputs=sliders
    )
    
    seed_input.change(
        fn=update_seed,
        inputs=seed_input,
        outputs=image_output
    )

    # 2. Image Projection
    btn_project.click(
        fn=run_projection,
        inputs=image_input,
        outputs=image_output
    ).then(
        fn=lambda: [0.0]*len(sliders), # Reset sliders on new image
        outputs=sliders
    )

    # 3. Sliders
    for s in sliders:
        s.change(
            fn=generate_wrapper,
            inputs=sliders,
            outputs=image_output
        )
        
    # 4. Reset
    btn_reset.click(
        fn=lambda: [0.0]*len(sliders),
        outputs=sliders
    )

if __name__ == "__main__":
    server_port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=server_port, share=False, debug=True, prevent_thread_lock=False)
