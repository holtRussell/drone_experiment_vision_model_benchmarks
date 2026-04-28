"""
Intermediate Representation Schema
All pipelines must normalize outputs to this format.
"""
from typing import Optional, Dict, Any
import json


def create_ir(
    cars: int = 0,
    pedestrians: int = 0,
    bicycles: int = 0,
    raw_description: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create standardized Intermediate Representation.
    
    Returns:
        Dict with standardized schema:
        {
            "cars": int,
            "pedestrians": int,
            "bicycles": int,
            "raw_description": str or None,
            "pipeline_name": str or None,
            "metadata": dict or None
        }
    """
    return {
        "cars": cars if cars is not None else 0,
        "pedestrians": pedestrians if pedestrians is not None else 0,
        "bicycles": bicycles if bicycles is not None else 0,
        "raw_description": raw_description,
        "pipeline_name": pipeline_name,
        "metadata": metadata or {}
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
