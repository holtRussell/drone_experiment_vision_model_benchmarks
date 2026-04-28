"""
Parser to convert VLM text outputs to Intermediate Representation
"""
import re
from typing import Dict, Any, Optional
from src.representation.schema import create_ir


def parse_vlm_response(
    vlm_response: str,
    pipeline_name: str = "vlm_direct"
) -> Dict[str, Any]:
    """
    Parse VLM text response to extract object counts.
    
    Args:
        vlm_response: Text response from VLM
        pipeline_name: Name of the pipeline for tracking
        
    Returns:
        Intermediate Representation dict
    """
    cars = 0
    pedestrians = 0
    bicycles = 0
    
    # Try to extract numbers using regex patterns
    # Pattern: "X cars", "cars: X", "number of cars: X", etc.
    
    # Cars
    car_patterns = [
        r'(\d+)\s*(?:cars?|vehicles?)',
        r'cars?[\s:]+(\d+)',
        r'(?:number of\s+)?cars?[\s:]*(?:is|was|are|were)?[\s:]*(\d+)',
    ]
    for pattern in car_patterns:
        match = re.search(pattern, vlm_response.lower())
        if match:
            cars = int(match.group(1))
            break
    
    # Pedestrians
    ped_patterns = [
        r'(\d+)\s*(?:pedestrians?|people|persons?)',
        r'pedestrians?[\s:]+(\d+)',
        r'(?:number of\s+)?pedestrians?[\s:]*(?:is|was|are|were)?[\s:]*(\d+)',
    ]
    for pattern in ped_patterns:
        match = re.search(pattern, vlm_response.lower())
        if match:
            pedestrians = int(match.group(1))
            break
    
    # Bicycles
    bike_patterns = [
        r'(\d+)\s*(?:bicycles?|bikes?|cyclists?)',
        r'bicycles?[\s:]+(\d+)',
        r'(?:number of\s+)?bicycles?[\s:]*(?:is|was|are|were)?[\s:]*(\d+)',
    ]
    for pattern in bike_patterns:
        match = re.search(pattern, vlm_response.lower())
        if match:
            bicycles = int(match.group(1))
            break
    
    return create_ir(
        cars=cars,
        pedestrians=pedestrians,
        bicycles=bicycles,
        raw_description=vlm_response,
        pipeline_name=pipeline_name,
        metadata={
            "parsing_method": "regex",
            "raw_response": vlm_response
        }
    )


def parse_vlm_json_response(
    vlm_response: str,
    pipeline_name: str = "vlm_direct"
) -> Dict[str, Any]:
    """
    Parse VLM response that returns JSON.
    """
    import json
    
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', vlm_response)
        if json_match:
            data = json.loads(json_match.group(0))
            
            return create_ir(
                cars=data.get('cars', 0),
                pedestrians=data.get('pedestrians', 0),
                bicycles=data.get('bicycles', 0),
                raw_description=vlm_response,
                pipeline_name=pipeline_name,
                metadata={
                    "parsing_method": "json",
                    "raw_response": vlm_response
                }
            )
    except:
        pass
    
    # Fall back to text parsing
    return parse_vlm_response(vlm_response, pipeline_name)
