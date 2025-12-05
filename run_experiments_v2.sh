#!/bin/bash

# Create log directory
mkdir -p output/logs

# Experiment 1: Smiling
echo "[$(date)] Starting Smiling experiment..."
python src/main_stylegan.py --attr Smiling --num-samples 50 --num-test-images 8 > output/logs/stylegan_smiling.log 2>&1
echo "[$(date)] Smiling experiment finished."

# Experiment 2: Male
echo "[$(date)] Starting Male experiment..."
python src/main_stylegan.py --attr Male --num-samples 50 --num-test-images 8 > output/logs/stylegan_male.log 2>&1
echo "[$(date)] Male experiment finished."

# Experiment 3: Young
echo "[$(date)] Starting Young experiment..."
python src/main_stylegan.py --attr Young --num-samples 50 --num-test-images 8 > output/logs/stylegan_young.log 2>&1
echo "[$(date)] Young experiment finished."

echo "[$(date)] All experiments completed."
