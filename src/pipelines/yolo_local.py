"""
YOLO Local Pipeline - Runs YOLO inference locally using Ultralytics
"""
import os
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.yolo_parser import parse_yolo_results


class YoloLocalPipeline(BasePipeline):
    """Local YOLO inference pipeline - Uses config-specified model"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_local", config)
        self.model = None
        self._model_specs = None
        
        # Model path priority: env var > config > default
        env_path = os.environ.get('YOLO_MODEL_PATH')
        if env_path:
            self.model_path = env_path
        else:
            default_path = "./models/yolov8m.pt"
            self.model_path = (config or {}).get('model', default_path)
    
    def _load_model(self):
        """Lazy-load YOLO model from configured path"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                
                # Resolve path (relative to project root)
                model_path = Path(self.model_path)
                if not model_path.is_absolute():
                    # Resolve relative to project root
                    project_root = Path(__file__).parent.parent.parent
                    model_path = project_root / self.model_path
                
                if not model_path.exists():
                    raise FileNotFoundError(f"YOLO model not found: {model_path}")
                
                print(f"Loading YOLO model: {model_path}")
                self.model = YOLO(str(model_path))
                
                # Extract full model specs
                self._model_specs = self._extract_model_specs(model_path)
                print(f"Model loaded. Classes: {self.model.names}")
                print(f"Model specs: {self._model_specs}")
                
            except ImportError:
                raise RuntimeError("Ultralytics not installed. Install with: pip install ultralytics")
    
    def _extract_model_specs(self, model_path: Path) -> Dict[str, Any]:
        """Extract full model specifications for IR metadata"""
        # Get file size
        file_size_bytes = model_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get class names from loaded model
        class_names = list(self.model.names.values()) if self.model else []
        
        # YOLOv8m known specs
        model_name = model_path.stem  # e.g., "yolov8m"
        
        specs = {
            "model_name": model_path.name,
            "model_path": str(model_path),
            "model_size_mb": round(file_size_mb, 2),
            "input_size": (640, 640),  # YOLO standard
            "framework": "ultralytics",
            "training_dataset": self._detect_training_dataset(),
            "num_classes": len(class_names),
            "classes": class_names,
            "architecture": "YOLOv8m" if "yolov8m" in model_path.name else "unknown",
            "params": "26M" if "yolov8m" in model_path.name else "unknown",
        }
        
        return specs
    
    def _detect_training_dataset(self) -> str:
        """Detect training dataset from class names"""
        if not self.model or not hasattr(self.model, 'names'):
            return "unknown"
        
        classes = set(v.lower() for v in self.model.names.values())
        
        # COCO detection (person, bicycle, car, etc.)
        coco_classes = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
        if coco_classes & classes:
            return "COCO"
        
        # VisDrone detection (pedestrian, bicycle, car, van, truck, etc.)
        visdrone_classes = {"pedestrian", "bicycle", "car", "van", "truck"}
        if visdrone_classes & classes:
            return "VisDrone"
        
        return "unknown"
    
    def get_model_specs(self) -> Dict[str, Any]:
        """Return model specs (load model first if needed)"""
        self._load_model()
        return self._model_specs
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image with local YOLO model.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation with enhanced metrics and model specs
        """
        self._load_model()
        
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model(image)
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse results to IR with image for resolution tracking
        ir = parse_yolo_results(results, pipeline_name=self.name, image=image)
        ir['metadata']['latency_ms'] = self._latency_ms
        
        # Add full model specs to IR
        if self._model_specs:
            ir['metadata']['model_specs'] = self._model_specs
        
        return ir