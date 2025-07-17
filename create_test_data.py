#!/usr/bin/env python3
"""
Create a minimal test dataset for MeanFlow training demonstration.
"""

import os
import json
import numpy as np
from PIL import Image
import torch

def create_test_dataset(data_dir="test_data", num_samples=100):
    """Create a minimal test dataset with synthetic images and features."""
    
    print(f"Creating test dataset in {data_dir} with {num_samples} samples...")
    
    # Create directories
    images_dir = os.path.join(data_dir, "images")
    features_dir = os.path.join(data_dir, "vae-sd")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Create synthetic images and VAE features
    labels = []
    feature_fnames = []
    
    for i in range(num_samples):
        # Generate synthetic image (256x256x3)
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        image_fname = f"img_{i:06d}.png"
        image_path = os.path.join(images_dir, image_fname)
        
        Image.fromarray(image).save(image_path)
        
        # Generate synthetic VAE features (typical VAE latent dimension: 8x32x32x4)
        # But we'll save it as 8-channel to include both mean and std
        vae_features = np.random.randn(8, 32, 32).astype(np.float32)
        feature_fname = f"img_{i:06d}.npy" 
        feature_path = os.path.join(features_dir, feature_fname)
        
        np.save(feature_path, vae_features)
        
        # Random class label (ImageNet has 1000 classes)
        label = np.random.randint(0, 1000)
        labels.append([feature_fname, label])
        feature_fnames.append(feature_fname)
    
    # Create dataset.json
    dataset_info = {
        "labels": labels
    }
    
    json_path = os.path.join(features_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Created {num_samples} synthetic samples")
    print(f"✓ Images saved to: {images_dir}")
    print(f"✓ VAE features saved to: {features_dir}")
    print(f"✓ Labels saved to: {json_path}")
    
    return data_dir

if __name__ == "__main__":
    create_test_dataset()
