"""
Parser to convert VLM text outputs to Intermediate Representation
"""
import re
import json
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
    try:
        return int(all_numbers[-1])
    except (ValueError, IndexError):
        return 0


def strip_system_reminders(text: str) -> str:
    """Remove <system-reminder> and everything after it"""
    if '<system-reminder>' in text:
        text = text.split('<system-reminder>')[0]
    return text.strip()


def parse_json_response(vlm_response: str) -> Optional[Dict[str, int]]:
    """
    Try to parse JSON from VLM response.
    
    Expected format:
    {
        Cars: 7,
        Pedestrians: 15,
        Bicycles: 8
    }
    
    Returns:
        Dict with keys 'cars', 'pedestrians', 'bicycles' or None if parsing fails
    """
    # Strip system reminders first
    vlm_response = strip_system_reminders(vlm_response)
    
    try:
        # Try to find JSON block (handle multi-line)
        json_match = re.search(r'\{[\s\S]*\}', vlm_response)
        if not json_match:
            return None
        
        json_str = json_match.group(0)
        
        # Normalize unquoted JSON keys like "Cars: 7" -> "\"Cars\": 7"
        # Handle patterns like "Cars: 7," -> "\"Cars\": 7,"
        json_str = re.sub(r'(\s)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        data = json.loads(json_str)
        
        # Normalize keys to lowercase for consistent access
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            if 'car' in key_lower and 'bicycle' not in key_lower:
                result['cars'] = int(value) if isinstance(value, (int, float)) else 0
            elif 'pedestrian' in key_lower or 'people' in key_lower:
                result['pedestrians'] = int(value) if isinstance(value, (int, float)) else 0
            elif 'bicycle' in key_lower or 'bike' in key_lower:
                result['bicycles'] = int(value) if isinstance(value, (int, float)) else 0
        
        return result if result else None
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parsing failed: {e}")
        print(f"  Response snippet: {vlm_response[:200]}...")
        return None
    except Exception as e:
        print(f"Warning: JSON parsing error: {e}")
        return None


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
    
    First tries JSON parsing, then falls back to regex patterns.
    
    Args:
        vlm_response: Text response from VLM
        pipeline_name: Name of the pipeline for tracking
        
    Returns:
        Intermediate Representation dict
    """
    # Try JSON parsing first (more reliable for structured responses)
    json_data = parse_json_response(vlm_response)
    
    if json_data:
        return create_ir(
            cars=json_data.get('cars', 0),
            pedestrians=json_data.get('pedestrians', 0),
            bicycles=json_data.get('bicycles', 0),
            raw_description=vlm_response,
            pipeline_name=pipeline_name,
            metadata={
                "parsing_method": "json",
                "raw_response": vlm_response,
                "json_parsed": json_data
            }
        )
    
    # Fall back to regex-based extraction
    cars = 0
    pedestrians = 0
    bicycles = 0
    
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
    Expected format:
        {
            Cars: 7,
            Pedestrians: 15,
            Bicycles: 8
        }
    
    If JSON parsing fails, prints warning and falls back to text parsing.
    """
    json_data = parse_json_response(vlm_response)
    
    if json_data:
        return create_ir(
            cars=json_data.get('cars', 0),
            pedestrians=json_data.get('pedestrians', 0),
            bicycles=json_data.get('bicycles', 0),
            raw_description=vlm_response,
            pipeline_name=pipeline_name,
            metadata={
                "parsing_method": "json",
                "raw_response": vlm_response,
                "json_parsed": json_data
            }
        )
    
    # Fall back to text parsing
    return parse_vlm_response(vlm_response, pipeline_name)
