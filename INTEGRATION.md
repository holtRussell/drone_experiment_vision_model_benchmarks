# Integration Guide: LLM Recommender Agent POC

This guide explains how to integrate with the LLM Recommender Agent POC for testing with custom datasets.

## Overview

The LLM Recommender Agent POC provides two key services:

1. **Object Detection Server (port 8099)** - MCP-based object detection using YOLOv8
2. **Image Server (port 8080)** - HTTP endpoint for serving images

## Setup

### Prerequisites
- Python 3.11 (ultralytics 8.2.10 requires Python 3.7-3.11)
- pip installed
- Access to the image dataset

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run servers from the image_detection directory
cd responses_api/image_detection
```

### Start Services

**Terminal 1 - Object Detection Server:**
```bash
python object_detection_server.py
```

**Terminal 2 - Image Server:**
```bash
python image_server.py
```

Both servers must be running for full functionality.

## Integration

### Object Detection Tool

The Object Detection Server exposes a Tool called `detect_objects_in_image` that can be called via MCP.

**Tool Details:**
- **Name:** `detect_objects_in_image`
- **Endpoint:** `http://localhost:8099/mcp`
- **Description:** Detects objects in images using YOLOv8 trained on VisDrone dataset

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image_location` | string | Filename of the image to analyze (must exist in `./images/` directory) |

**Response Format:**
```json
{
  "text": "Detected object \"person\" in image with confidence 0.85. Detected object \"car\" in image with confidence 0.72.",
  "image_url": "/absolute/path/to/outputs/outputs/filename.jpg"
}
```

**Example MCP Call:**
```bash
curl -X POST http://localhost:8099/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "detect_objects_in_image",
      "arguments": {
        "image_location": "your_image.jpg"
      }
    }
  }'
```

**MCP Notification Response:**
```json
{
  "method": "notifications/progress",
  "params": {
    "progressToken": "token",
    "progress": 0.5,
    "message": "Processing..."
  }
}
```

### Image Serving

The Image Server provides HTTP access to images in the `./images/` directory.

**Endpoint:** `http://localhost:8080/<image_filename>`

**Example:**
```bash
# View image in browser
open http://localhost:8080/your_image.jpg

# Download via curl
curl -o downloaded.jpg http://localhost:8080/your_image.jpg

# Get image info
curl -I http://localhost:8080/your_image.jpg
```

### Output Images

Detection output images are saved to `./outputs/outputs/` with the same filename as the input. The `image_url` in the response points to this location.

## Testing with Custom Dataset

### Preparing Your Dataset

1. Place your test images in the `./images/` directory:
   ```
   responses_api/image_detection/images/
   ├── image1.jpg
   ├── image2.jpg
   └── your_custom_image.jpg
   ```

2. Ensure image formats are supported: JPEG, PNG, BMP

### Running Object Detection

**Via MCP (recommended for LLM integration):**

```python
import requests
import json

def detect_objects(image_filename):
    """Call the object detection tool via MCP"""
    url = "http://localhost:8099/mcp"
    payload = {
        "method": "tools/call",
        "params": {
            "name": "detect_objects_in_image",
            "arguments": {
                "image_location": image_filename
            }
        }
    }
    response = requests.post(url, json=payload)
    return response.json()
```

**Via curl (command line testing):**

```bash
curl -X POST http://localhost:8099/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "detect_objects_in_image",
      "arguments": {
        "image_location": "test1.jpg"
      }
    }
  }'
```

### Expected Outputs

**Success Response:**
```json
{
  "text": "Detected object \"person\" in image with confidence 0.92. Detected object \"car\" in image with confidence 0.87.",
  "image_url": "/Users/holtrussell/Downloads/LLM-Recommender-Agent-POC/responses_api/image_detection/outputs/outputs/test1.jpg"
}
```

**No Objects Detected:**
```json
{
  "text": "No objects detected in image.",
  "image_url": "/Users/holtrussell/Downloads/LLM-Recommender-Agent-POC/responses_api/image_detection/outputs/outputs/test1.jpg"
}
```

**Error Response:**
```json
{
  "error": "Image file not found: your_file.jpg"
}
```

## Supported Detection Classes

The YOLOv8 model (Visdrone_yolov8s.pt) detects the following object classes from the VisDrone dataset:
1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor

## Troubleshooting

### Port Already in Use
If port 8099 or 8080 is already in use, you can modify the port in:
- `object_detection_server.py`: Line 18 `server = FastMCP(..., port=8099)`
- `image_server.py`: Line 15 `app.run(host='0.0.0.0', port=8080)`

### Model Not Loading
If you see weight loading errors, ensure:
1. You have permissions to read `Visdrone_yolov8s.pt`
2. The file is not corrupted (should be ~23MB)

### No Objects Detected
- Adjust confidence threshold in `object_detection_server.py` line 27: `conf=0.2`
- Lower value (e.g., 0.1) = more detections, higher value (e.g., 0.5) = fewer but more confident detections

### Images Not Served
- Verify images are in `./images/` directory
- Check file permissions
- Ensure image_server.py is running on port 8080

## API Reference

### Object Detection Server (MCP)

**Method:** `tools/call`

**Parameters:**
```json
{
  "name": "detect_objects_in_image",
  "arguments": {
    "image_location": "string"
  }
}
```

### Image Server (HTTP)

**Method:** `GET`

**Endpoint:** `http://localhost:8080/<image_filename>`

**Response:** Image bytes with `Content-Type: image/jpeg`
