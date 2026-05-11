"""
Parser to convert VLM text outputs to Intermediate Representation
"""
import re
from typing import Dict, Any, Optional, List
from src.representation.schema import create_ir


def extract_count_from_text(text: str, obj_type: str) -> int:
    """Extract count for specific object type from text"""
    text_lower = text.lower()
    
    # Find all numbers in the text
    all_numbers = re.findall(r'\d+', text)
    if not all_numbers:
        return 0
    
    # For mock mode, return the last number found (usually the actual count)
    # In real mode, this would be more sophisticated
    try:
        return int(all_numbers[-1])  # Return last number found
    except (ValueError, IndexError):
        return 0
    
    # For mock mode, just return the first number found
    # In real mode, this would be more sophisticated
    try:
        return int(numbers[0])
    except (ValueError, IndexError):
        return 0


def parse_atomic_responses(
    atomic_responses: Dict[str, str],
    composite_response: str,
    pipeline_name: str
) -> Dict[str, Any]:
    """
    Parse BOTH atomic and composite VLM responses for counts.
    
    Args:
        atomic_responses: Dict with keys 'cars', 'pedestrians', 'bicycles'
                          and values = LLM text responses
        composite_response: Full composite response text
        pipeline_name: Pipeline name for tracking
    
    Returns:
        Intermediate Representation with counts from both sources
    """
    # Initialize counts
    cars = 0
    pedestrians = 0
    bicycles = 0
    
    atomic_counts = {}
    
    # Parse atomic responses (more reliable for counts)
    for obj_type, response in atomic_responses.items():
        count = extract_count_from_text(response, obj_type)
        atomic_counts[obj_type] = count
        
        if obj_type == 'cars':
            cars = count
        elif obj_type == 'pedestrians':
            pedestrians = count
        elif obj_type == 'bicycles':
            bicycles = count
    
    # Fallback: Parse composite response if atomic failed
    if cars == 0 or pedestrians == 0 or bicycles == 0:
        composite_counts = {
            'cars': extract_count_from_text(composite_response, 'cars'),
            'pedestrians': extract_count_from_text(composite_response, 'pedestrians'),
            'bicycles': extract_count_from_text(composite_response, 'bicycles'),
        }
        
        cars = cars or composite_counts['cars']
        pedestrians = pedestrians or composite_counts['pedestrians']
        bicycles = bicycles or composite_counts['bicycles']
    
    return create_ir(
        cars=cars,
        pedestrians=pedestrians,
        bicycles=bicycles,
        raw_description=composite_response,
        pipeline_name=pipeline_name,
        metadata={
            "parsing_method": "atomic_then_composite",
            "atomic_responses": atomic_responses,
            "composite_response": composite_response,
            "atomic_counts": atomic_counts,
            "composite_counts": {
                'cars': extract_count_from_text(composite_response, 'cars'),
                'pedestrians': extract_count_from_text(composite_response, 'pedestrians'),
                'bicycles': extract_count_from_text(composite_response, 'bicycles'),
            }
        }
    )


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
