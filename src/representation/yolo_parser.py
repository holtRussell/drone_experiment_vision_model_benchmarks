"""
Parser to convert YOLO detection results to Intermediate Representation
"""
from typing import Dict, Any, List
from src.representation.schema import create_ir


# YOLO class names (COCO dataset - common classes for VisDrone related objects)
# Adjust based on your YOLO model's training
YOLO_CLASS_MAP = {
    0: "person",    # Usually class 0 in COCO
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    # Add more as needed
}


def parse_yolo_results(
    yolo_results,
    pipeline_name: str = "yolo_local"
) -> Dict[str, Any]:
    """
    Parse YOLO results to Intermediate Representation.
    
    Args:
        yolo_results: Ultralytics YOLO results object
        pipeline_name: Name of the pipeline for tracking
        
    Returns:
        Intermediate Representation dict
    """
    cars = 0
    pedestrians = 0
    bicycles = 0
    
    detections = []
    
    # Handle both single result and list of results
    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]
    
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                
                class_name = YOLO_CLASS_MAP.get(cls_id, f"unknown_{cls_id}")
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": box.xyxy.tolist()[0] if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                })
                
                if class_name == "person":
                    pedestrians += 1
                elif class_name == "car":
                    cars += 1
                elif class_name == "bicycle" or class_name == "motorcycle":
                    bicycles += 1
    
    return create_ir(
        cars=cars,
        pedestrians=pedestrians,
        bicycles=bicycles,
        raw_description=f"Detected {len(detections)} objects: {cars} cars, {pedestrians} pedestrians, {bicycles} bicycles",
        pipeline_name=pipeline_name,
        metadata={
            "detections": detections,
            "total_detections": len(detections)
        }
    )


def parse_yolo_detections_from_mcp(
    mcp_response: Dict[str, Any],
    pipeline_name: str = "yolo_mcp"
) -> Dict[str, Any]:
    """
    Parse YOLO detections from MCP server response.
    
    Expected MCP response format:
    {
        "detections": [
            {"class": "car", "confidence": 0.95, "bbox": [x1, y1, x2, y2]},
            ...
        ]
    }
    """
    cars = 0
    pedestrians = 0
    bicycles = 0
    
    detections = mcp_response.get("detections", [])
    
    for det in detections:
        class_name = det.get("class", "").lower()
        
        if class_name in ["person", "pedestrian"]:
            pedestrians += 1
        elif class_name in ["car", "van", "truck", "bus"]:
            cars += 1
        elif class_name in ["bicycle", "bike", "motorcycle"]:
            bicycles += 1
    
    return create_ir(
        cars=cars,
        pedestrians=pedestrians,
        bicycles=bicycles,
        raw_description=f"Detected {len(detections)} objects via MCP",
        pipeline_name=pipeline_name,
        metadata={
            "detections": detections,
            "total_detections": len(detections)
        }
    )
