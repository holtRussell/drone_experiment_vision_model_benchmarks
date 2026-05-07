"""
Intermediate Representation Schema
All pipelines must normalize outputs to this format.
"""
from typing import Optional, Dict, Any, List
import json


def create_ir(
    cars: int = 0,
    pedestrians: int = 0,
    bicycles: int = 0,
    raw_description: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    metadata: Optional[Dict] = None,
    confidence_scores: Optional[List[float]] = None,
    avg_confidence_per_class: Optional[Dict[str, float]] = None,
    image_resolution: Optional[tuple] = None,
    model_input_size: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    Create standardized Intermediate Representation.
    
    Args:
        cars: Number of cars detected
        pedestrians: Number of pedestrians detected
        bicycles: Number of bicycles detected
        raw_description: Text description of detection
        pipeline_name: Name of the pipeline that generated this IR
        metadata: Additional metadata dict
        confidence_scores: List of all detection confidence scores
        avg_confidence_per_class: Dict with per-class average confidence
        image_resolution: (width, height) of input image
        model_input_size: (width, height) the model processed
        
    Returns:
        Dict with standardized schema
    """
    if metadata is None:
        metadata = {}
    
    # Add vision-specific metrics to metadata
    if confidence_scores is not None:
        metadata['confidence_scores'] = confidence_scores
    if avg_confidence_per_class is not None:
        metadata['avg_confidence_per_class'] = avg_confidence_per_class
    if image_resolution is not None:
        metadata['image_resolution'] = image_resolution
    if model_input_size is not None:
        metadata['model_input_size'] = model_input_size
    
    return {
        "cars": cars if cars is not None else 0,
        "pedestrians": pedestrians if pedestrians is not None else 0,
        "bicycles": bicycles if bicycles is not None else 0,
        "raw_description": raw_description,
        "pipeline_name": pipeline_name,
        "metadata": metadata
    }


def ir_to_json(ir: Dict[str, Any]) -> str:
    """Convert IR to JSON string"""
    return json.dumps(ir, indent=2)


def ir_from_json(json_str: str) -> Dict[str, Any]:
    """Parse IR from JSON string"""
    return json.loads(json_str)


def validate_ir(ir: Dict[str, Any]) -> bool:
    """Validate IR has required fields"""
    required_fields = ["cars", "pedestrians", "bicycles"]
    return all(field in ir for field in required_fields)
