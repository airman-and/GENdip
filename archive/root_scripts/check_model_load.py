import sys
import os
import pickle
import torch

# Add stylegan2_ada_pytorch to path
stylegan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

def check_model():
    model_path = 'checkpoints/stylegan2_ada/ffhq.pkl'
    print(f"Checking model: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
        
    try:
        print("Loading model...")
        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema']
        print("Model loaded successfully!")
        print(f"Generator class: {type(G)}")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
