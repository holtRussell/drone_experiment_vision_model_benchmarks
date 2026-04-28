"""
YOLO Local Pipeline - Runs YOLO inference locally using Ultralytics
"""
import time
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.yolo_parser import parse_yolo_results


class YoloLocalPipeline(BasePipeline):
    """Local YOLO inference pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_local", config)
        self.model = None
        self.model_name = (config or {}).get('model', 'yolov8n.pt')
    
    def _load_model(self):
        """Lazy-load YOLO model"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
            except ImportError:
                raise RuntimeError("Ultralytics not installed. Install with: pip install ultralytics")
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image with local YOLO model.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation
        """
        self._load_model()
        
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model(image)
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse results to IR
        ir = parse_yolo_results(results, pipeline_name=self.name)
        ir['metadata']['latency_ms'] = self._latency_ms
        
        return ir
