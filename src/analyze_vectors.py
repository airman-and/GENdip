import torch
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def load_vector(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None
    return np.load(path)

def main():
    base_dir = "output/vectors"
    
    # Load vectors
    vec_glasses = load_vector(os.path.join(base_dir, "vector_Eyeglasses.npy"))
    vec_glasses_bal = load_vector(os.path.join(base_dir, "vector_Eyeglasses_balanced.npy"))
    vec_male = load_vector(os.path.join(base_dir, "vector_Male.npy"))
    vec_young = load_vector(os.path.join(base_dir, "vector_Young.npy"))
    
    if any(v is None for v in [vec_glasses, vec_glasses_bal, vec_male, vec_young]):
        return

    # Reshape for sklearn (1, 512)
    vec_glasses = vec_glasses.reshape(1, -1)
    vec_glasses_bal = vec_glasses_bal.reshape(1, -1)
    vec_male = vec_male.reshape(1, -1)
    vec_young = vec_young.reshape(1, -1)
    
    # Calculate Similarities
    print("--- Vector Similarity Analysis ---")
    
    # 1. Similarity between Original and Debiased
    sim_orig_bal = cosine_similarity(vec_glasses, vec_glasses_bal)[0][0]
    print(f"Similarity (Original vs Debiased): {sim_orig_bal:.4f}")
    
    # 2. Entanglement with Male
    sim_orig_male = cosine_similarity(vec_glasses, vec_male)[0][0]
    sim_bal_male = cosine_similarity(vec_glasses_bal, vec_male)[0][0]
    print(f"Correlation with Male: Original={sim_orig_male:.4f} -> Debiased={sim_bal_male:.4f}")
    
    # 3. Entanglement with Young
    sim_orig_young = cosine_similarity(vec_glasses, vec_young)[0][0]
    sim_bal_young = cosine_similarity(vec_glasses_bal, vec_young)[0][0]
    print(f"Correlation with Young: Original={sim_orig_young:.4f} -> Debiased={sim_bal_young:.4f}")

if __name__ == "__main__":
    main()
