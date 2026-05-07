"""
YOLO Local Pipeline - Runs YOLO inference locally using Ultralytics
"""
import time
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.yolo_parser import parse_yolo_results


class YoloLocalPipeline(BasePipeline):
    """Local YOLO inference pipeline - Uses VisDrone-trained model"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_local", config)
        self.model = None
        # Use YOLOv8n (will auto-download) - can be replaced with VisDrone-trained model
        self.model_name = (config or {}).get('model', 'yolov8n.pt')
    
    def _load_model(self):
        """Lazy-load YOLO model (auto-downloads from Ultralytics Hub if needed)"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                print(f"Loading YOLO model: {self.model_name}")
                
                # Handle HuggingFace model format (e.g., mshamrai/yolov8n-visdrone)
                if '/' in self.model_name and not self.model_name.endswith('.pt'):
                    # Try to download from HuggingFace to a local file
                    try:
                        from huggingface_hub import hf_hub_download
                        local_path = hf_hub_download(
                            repo_id=self.model_name.split('/')[0] + '/' + self.model_name.split('/')[1],
                            filename='yolov8n-visdrone.pt',
                            cache_dir='./models'
                        )
                        self.model = YOLO(local_path)
                    except Exception as e:
                        print(f"Could not load from HuggingFace: {e}")
                        print("Falling back to yolov8n.pt (will download if needed)")
                        self.model = YOLO('yolov8n.pt')
                else:
                    # Standard ultralytics model (auto-downloads if needed)
                    self.model = YOLO(self.model_name)
                
                print(f"Model loaded. Classes: {self.model.names}")
            except ImportError:
                raise RuntimeError("Ultralytics not installed. Install with: pip install ultralytics")
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image with local YOLO model.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation with enhanced metrics
        """
        self._load_model()
        
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model(image)
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse results to IR with image for resolution tracking
        ir = parse_yolo_results(results, pipeline_name=self.name, image=image)
        ir['metadata']['latency_ms'] = self._latency_ms
        
        return ir
