import random
from typing import List, Dict
from src.data.loader import VisDroneLoader
from src.data.ground_truth import VisDroneGroundTruth


class DatasetSampler:
    """Sample images from VisDrone dataset with stratification"""
    
    def __init__(self, dataset_path: str, seed: int = 42):
        self.loader = VisDroneLoader(dataset_path)
        self.ground_truth = VisDroneGroundTruth()
        self.seed = seed
        random.seed(seed)
    
    def sample_by_density(self, num_samples: int, annotations_dir: str) -> List[Dict]:
        """
        Sample images with varied scene densities.
        Stratifies by: low (0-5 objects), medium (6-15), high (16+)
        """
        all_images = list(self.loader)
        
        # Categorize by density
        low_density = []
        medium_density = []
        high_density = []
        
        for img_data in all_images:
            if img_data.get("annotation_path") and Path(img_data["annotation_path"]).exists():
                counts = self.ground_truth.get_ground_truth(
                    img_data["image_id"], 
                    str(Path(img_data["annotation_path"]).parent)
                )
                total = sum(counts.values())
                
                if total <= 5:
                    low_density.append(img_data)
                elif total <= 15:
                    medium_density.append(img_data)
                else:
                    high_density.append(img_data)
            else:
                # No annotation, treat as low density
                low_density.append(img_data)
        
        # Sample proportionally
        num_low = max(1, int(num_samples * 0.3))
        num_medium = max(1, int(num_samples * 0.4))
        num_high = num_samples - num_low - num_medium
        
        sampled = []
        sampled.extend(random.sample(low_density, min(num_low, len(low_density))))
        sampled.extend(random.sample(medium_density, min(num_medium, len(medium_density))))
        sampled.extend(random.sample(high_density, min(num_high, len(high_density))))
        
        return sampled[:num_samples]
    
    def get_sample(self, num_samples: int) -> List[Dict]:
        """Get a simple random sample"""
        all_images = list(self.loader)
        return random.sample(all_images, min(num_samples, len(all_images)))
