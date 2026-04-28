"""
YOLO via MCP Server Pipeline - Placeholder for MCP integration
We'll integrate the actual MCP server code when provided.
"""
import time
import requests
from typing import Dict, Any
from PIL import Image
import io
import base64
from src.pipelines.base import BasePipeline
from src.representation.yolo_parser import parse_yolo_detections_from_mcp


class YoloMcpPipeline(BasePipeline):
    """YOLO inference via MCP server"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_mcp", config)
        self.mcp_url = (config or {}).get('mcp_url', 'http://localhost:8080')
        self.endpoint = (config or {}).get('endpoint', '/detect')
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image via MCP server running YOLO.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation
        """
        start_time = time.perf_counter()
        
        # Prepare request to MCP server
        image_b64 = self._encode_image(image)
        
        payload = {
            "image": image_b64,
            "format": "jpeg"
        }
        
        try:
            response = requests.post(
                f"{self.mcp_url}{self.endpoint}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            mcp_result = response.json()
            
        except requests.exceptions.RequestException as e:
            # Return empty IR on error
            return {
                "cars": 0,
                "pedestrians": 0,
                "bicycles": 0,
                "raw_description": f"MCP Error: {str(e)}",
                "pipeline_name": self.name,
                "metadata": {"error": str(e)}
            }
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse MCP response to IR
        ir = parse_yolo_detections_from_mcp(mcp_result, pipeline_name=self.name)
        ir['metadata']['latency_ms'] = self._latency_ms
        ir['metadata']['mcp_url'] = self.mcp_url
        
        return ir
