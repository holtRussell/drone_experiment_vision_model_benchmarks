"""
Parser to convert YOLO detection results to Intermediate Representation
Supports both COCO and VisDrone-trained YOLO models
"""
from typing import Dict, Any, List
from src.representation.schema import create_ir


# VisDrone dataset class mapping (10 classes)
# From: https://github.com/VisDrone/VisDrone-Dataset
VISDRONE_CLASS_MAP = {
    0: "pedestrian",   # 1 in VisDrone annotation
    1: "pedestrian",   # 2: people
    2: "bicycle",       # 3: bicycle
    3: "car",           # 4: car
    4: "car",           # 5: van (count as car)
    5: "car",           # 6: truck (count as car)
    6: "bicycle",       # 7: tricycle (count as bicycle)
    7: "bicycle",       # 8: awning-tricycle (count as bicycle)
    8: "car",           # 9: bus (count as car)
    9: "bicycle",       # 10: motor (count as bicycle)
}

# COCO class mapping (for reference)
COCO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
}


def detect_model_type(model_names: dict) -> str:
    """Detect if model is trained on VisDrone or COCO based on class names"""
    if not model_names:
        return "unknown"
    
    # Check for VisDrone-specific classes
    visdrone_classes = {"pedestrian", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}
    model_classes = set(model_names.values())
    
    # If any VisDrone-specific class exists
    if model_classes & visdrone_classes:
        return "visdrone"
    
    return "coco"


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
    
    # Detect model type from first result
    model_type = "coco"  # default
    if yolo_results and hasattr(yolo_results[0], 'names'):
        model_type = detect_model_type(yolo_results[0].names)
    
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                
                # Get class name from model
                if hasattr(result, 'names') and cls_id in result.names:
                    class_name = result.names[cls_id].lower()
                else:
                    class_name = f"unknown_{cls_id}"
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": box.xyxy.tolist()[0] if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                })
                
                # Map to our three categories
                class_name_lower = class_name.lower()
                
                if model_type == "visdrone":
                    # VisDrone model - direct mapping
                    if 'pedestrian' in class_name_lower or 'people' in class_name_lower:
                        pedestrians += 1
                    elif 'car' in class_name_lower or 'van' in class_name_lower or 'truck' in class_name_lower or 'bus' in class_name_lower:
                        cars += 1
                    elif 'bicycle' in class_name_lower or 'tricycle' in class_name_lower or 'motor' in class_name_lower:
                        bicycles += 1
                else:
                    # COCO model
                    if 'person' in class_name_lower:
                        pedestrians += 1
                    elif 'car' in class_name_lower or 'truck' in class_name_lower or 'bus' in class_name_lower:
                        cars += 1
                    elif 'bicycle' in class_name_lower or 'motorcycle' in class_name_lower:
                        bicycles += 1
    
    return create_ir(
        cars=cars,
        pedestrians=pedestrians,
        bicycles=bicycles,
        raw_description=f"Detected {len(detections)} objects: {cars} cars, {pedestrians} pedestrians, {bicycles} bicycles",
        pipeline_name=pipeline_name,
        metadata={
            "detections": detections,
            "total_detections": len(detections),
            "model_type": model_type
        }
    )


def parse_yolo_detections_from_mcp(
    mcp_response: Dict[str, Any],
    pipeline_name: str = "yolo_mcp"
) -> Dict[str, Any]:
    """
    Parse YOLO detections from MCP server response.
    Handles both COCO and VisDrone class names.
    
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
        
        # Map to our three categories (handle both COCO and Visdrone)
        if class_name in ["person", "pedestrian", "people"]:
            pedestrians += 1
        elif class_name in ["car", "van", "truck", "bus"]:
            cars += 1
        elif class_name in ["bicycle", "bike", "motorcycle", "tricycle", "awning-tricycle", "motor"]:
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
