#!/usr/bin/env python3
"""
Script to download all required datasets for the SVM project.
"""

import sys
import os

from src.utils.data_loader import DataLoader

def main():
    """Download all datasets."""
    print("Initializing DataLoader...")
    loader = DataLoader()
    
    print("\n=== Downloading Datasets ===")
    
    # Download classification datasets
    print("\n1. Heart Disease Dataset (Classification)")
    try:
        heart_data = loader.load_heart_disease()
        print(f"✓ Heart Disease data loaded: {heart_data.shape}")
    except Exception as e:
        print(f"✗ Error loading heart disease data: {e}")
    
    print("\n2. BBC News Dataset (Classification)")
    try:
        bbc_data = loader.load_bbc_news()
        print(f"✓ BBC News data loaded: {bbc_data.shape}")
    except Exception as e:
        print(f"✗ Error loading BBC news data: {e}")
    
    # Download regression datasets
    print("\n3. Wine Quality Dataset (Regression)")
    try:
        wine_data = loader.load_wine_quality()
        print(f"✓ Wine Quality data loaded: {wine_data.shape}")
    except Exception as e:
        print(f"✗ Error loading wine quality data: {e}")
    
    print("\n4. California Housing Dataset (Regression)")
    try:
        housing_data = loader.load_california_housing()
        print(f"✓ California Housing data loaded: {housing_data.shape}")
    except Exception as e:
        print(f"✗ Error loading california housing data: {e}")
    
    print("\n=== Dataset Download Complete ===")
    
    # List downloaded files
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"\nFiles in {data_dir}:")
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size:,} bytes)")

if __name__ == "__main__":
    main()
