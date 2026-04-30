from pathlib import Path
from typing import List, Dict, Optional, Iterator
from PIL import Image


class VisDroneLoader:
    """Load and iterate over VisDrone dataset images"""
    
    def __init__(self, dataset_path: str, max_dimension: int = 640):
        self.dataset_path = Path(dataset_path)
        self.max_dimension = max_dimension
        self.images_dir = self.dataset_path / "images"
        self.annotations_dir = self.dataset_path / "annotations"
        
    def load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        
        # Resize to max_dimension while maintaining aspect ratio
        if max(img.size) > self.max_dimension:
            ratio = self.max_dimension / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
    
    def get_image_list(self) -> List[Path]:
        """Get list of all image paths"""
        if not self.images_dir.exists():
            return []
        
        return sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over dataset, yielding image data"""
        image_paths = self.get_image_list()
        
        for img_path in image_paths:
            image_id = img_path.stem  # filename without extension
            
            yield {
                "image_id": image_id,
                "image_path": str(img_path),
                "annotation_path": str(self.annotations_dir / f"{image_id}.txt") if self.annotations_dir.exists() else None
            }
    
    def __len__(self) -> int:
        return len(self.get_image_list())
