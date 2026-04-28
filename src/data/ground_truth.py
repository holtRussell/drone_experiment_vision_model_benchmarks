from pathlib import Path
from typing import Dict, Optional
import pandas as pd


class VisDroneGroundTruth:
    """
    Parse VisDrone annotations to extract object counts.
    
    VisDrone annotation format (per line):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    Object categories:
    1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van, 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor
    """
    
    CATEGORY_MAP = {
        1: "pedestrian",  # pedestrian
        2: "pedestrian",  # people (count as pedestrian)
        3: "bicycle",
        4: "car",
        5: "car",         # van (count as car for simplicity)
        6: "car",         # truck (count as car for simplicity)
        7: "bicycle",     # tricycle (count as bicycle)
        8: "bicycle",     # awning-tricycle (count as bicycle)
        9: "car",         # bus (count as car)
        10: "bicycle",    # motor (count as bicycle)
    }
    
    def __init__(self, annotation_path: Optional[str] = None):
        self.annotation_path = Path(annotation_path) if annotation_path else None
    
    def parse_annotation(self, annotation_path: Optional[str] = None) -> Dict[str, int]:
        """Parse annotation file and return object counts"""
        path = Path(annotation_path) if annotation_path else self.annotation_path
        
        if not path or not path.exists():
            return {"cars": 0, "pedestrians": 0, "bicycles": 0}
        
        counts = {"cars": 0, "pedestrians": 0, "bicycles": 0}
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                try:
                    category_id = int(parts[5])
                    category = self.CATEGORY_MAP.get(category_id)
                    
                    if category == "car":
                        counts["cars"] += 1
                    elif category == "pedestrian":
                        counts["pedestrians"] += 1
                    elif category == "bicycle":
                        counts["bicycles"] += 1
                except (ValueError, IndexError):
                    continue
        
        return counts
    
    def get_ground_truth(self, image_id: str, annotations_dir: str) -> Dict[str, int]:
        """Get ground truth counts for an image"""
        annotation_path = Path(annotations_dir) / f"{image_id}.txt"
        return self.parse_annotation(str(annotation_path))
