"""
Download VisDrone dataset.
Dataset available from: https://github.com/VisDrone/VisDrone-Dataset
"""
import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


VISDRONE_URLS = {
    "VisDrone2019-DET-train": "https://github.com/VisDrone/VisDrone-Dataset/raw/master/Release/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val": "https://github.com/VisDrone/VisDrone-Dataset/raw/master/Release/VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev": "https://github.com/VisDrone/VisDrone-Dataset/raw/master/Release/VisDrone2019-DET-test-dev.zip",
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {zip_path.name}")


def download_visdrone(data_dir: str = "data/visdrone", subset: str = "train"):
    """
    Download and extract VisDrone dataset.
    
    Args:
        data_dir: Directory to store dataset
        subset: Which subset to download ('train', 'val', 'test', or 'all')
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    subsets = ["train", "val", "test"] if subset == "all" else [subset]
    
    key_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev"
    }
    
    for sub in subsets:
        key = key_map[sub]
        url = VISDRONE_URLS[key]
        zip_path = data_path / f"{key}.zip"
        
        print(f"\nDownloading {key}...")
        
        if not zip_path.exists():
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"Error downloading {key}: {e}")
                print(f"Please manually download from: {url}")
                print(f"And place in: {zip_path}")
                continue
        
        if zip_path.exists():
            print(f"Extracting {key}...")
            extract_zip(zip_path, data_path)
            print(f"Cleaning up {zip_path.name}...")
            zip_path.unlink()
    
    print(f"\nDataset ready at: {data_path.absolute()}")
    print(f"Please organize images in {data_path}/images/ and annotations in {data_path}/annotations/")


if __name__ == "__main__":
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else "train"
    download_visdrone(subset=subset)
