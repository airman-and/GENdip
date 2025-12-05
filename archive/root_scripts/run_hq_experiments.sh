#!/bin/bash
mkdir -p output/logs

# Function to run experiment
run_exp() {
    attr=$1
    echo "[$(date)] Starting $attr experiment (HQ)..."
    python src/main_stylegan.py --attr "$attr" --num-samples 200 --num-test-images 8 > "output/logs/stylegan_${attr}_hq.log" 2>&1
    echo "[$(date)] $attr experiment finished."
}

# Run experiments sequentially
run_exp "Male"
run_exp "Blond_Hair"
run_exp "Eyeglasses"
run_exp "Bangs"
run_exp "Pale_Skin"
run_exp "Chubby"

echo "[$(date)] All HQ experiments completed."
