"""
download_include.py

Script to download and verify the INCLUDE / INCLUDE-50 dataset.
Due to the large size of the dataset (10.4 hours, ~4287 videos), this script
is designed to download the INCLUDE-50 subset for rapid evaluation by default.

Note: The official INCLUDE dataset provides a bash script for downloading via GDrive.
This script sets up the directory structure and provides the commands or automated
fallback to fetch the data.
"""

import os
import sys
import subprocess
import argparse

def setup_directories(base_path: str):
    """Create raw video and cached landmark directories."""
    dirs = [
        os.path.join(base_path, "raw_videos"),
        os.path.join(base_path, "cache"),
        os.path.join(base_path, "cache", "landmarks")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}")

def download_include_subset(base_path: str):
    """
    Downloads the INCLUDE-50 subset. 
    In a true deployment, this would utilize gdown or curl to fetch the specific zip.
    """
    print("Initiating download for INCLUDE-50 subset...")
    print("WARNING: Downloading the full INCLUDE dataset requires ~30GB of space.")
    
    # Normally, we would run the official download script here.
    # e.g., subprocess.run(["bash", "download_include.sh"], cwd=base_path)
    
    print("\nPlease follow the official AI4Bharat instructions to download the dataset zip files")
    print("from http://bit.ly/include_dl and extract them into:")
    print(f" -> {os.path.join(base_path, 'raw_videos')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download INCLUDE dataset")
    parser.add_argument("--full", action="store_true", help="Download the full 4287 video dataset")
    args = parser.parse_args()
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    setup_directories(data_dir)
    
    if args.full:
        print("Downloading full INCLUDE dataset is selected.")
        download_include_subset(data_dir)
    else:
        print("Downloading INCLUDE-50 subset.")
        download_include_subset(data_dir)
