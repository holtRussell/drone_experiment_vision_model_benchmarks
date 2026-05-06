# Patch torch.load to use weights_only=False before ultralytics imports it
import torch
_original_load = torch.load
def _patched_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = _patched_load

from mcp.server.fastmcp import FastMCP
from mcp import types
import sys
from typing import Annotated
from pydantic import Field
from ultralytics import YOLO

# Create the server
server = FastMCP("Agent Tools Server", host='0.0.0.0', port=8099, stateless_http=True, json_response=True)

# Use YOLO's built-in model loading
img_model = YOLO("./Visdrone_yolov8s.pt")

@server.tool(name="detect_objects_in_image", description="""Calls an object detection model to identify objects in an image. Returns a JSON list of objects detected in the image with class names and confidence scores.""")
def detect_objects_in_image(image_location: Annotated[str, Field(description='Absolute filepath to image to run detection model on')]):
    """Calls an object detection model to identify objects in an image. Returns structured JSON with detections."""

    import os
    from pathlib import Path

    # Save outputs to project root detection_outputs folder
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "detection_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use absolute path directly (no prepending ./images/)
    results = img_model.predict(source=image_location, conf=0.2, save=True, project=str(output_dir), name='outputs', exist_ok=True)

    detections = []
    result_info = results[0]

    for i in range(len(result_info.boxes.cls)):
        name = img_model.names[int(result_info.boxes.cls[i])]
        conf = round(result_info.boxes.conf[i].item(), 3)
        bbox = result_info.boxes.xyxy[i].tolist()
        detections.append({
            "class": name,
            "confidence": conf,
            "bbox": bbox
        })

    image_url = os.path.join(result_info.save_dir, os.path.basename(image_location))

    return {
        "detections": detections,
        "total_detections": len(detections),
        "image_url": image_url
    }

# Run the server
if __name__ == "__main__":
    server.run(transport="streamable-http")
