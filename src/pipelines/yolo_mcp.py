"""
YOLO via gRPC Service Pipeline - Uses TypeFly's YOLO gRPC service
"""
import time
import io
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.yolo_parser import parse_yolo_detections_from_mcp
from src.serving.grpc_client import YoloGrpcClient


class YoloMcpPipeline(BasePipeline):
    """YOLO inference via gRPC service (from TypeFly)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_mcp", config)
        self.host = (config or {}).get('host', 'localhost')
        self.port = (config or {}).get('port', 50050)
        self.conf = (config or {}).get('conf', 0.3)
        self.client = None
    
    def _get_client(self) -> YoloGrpcClient:
        """Get or create gRPC client"""
        if self.client is None:
            self.client = YoloGrpcClient(host=self.host, port=self.port)
        return self.client
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image via gRPC YOLO service.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation
        """
        start_time = time.perf_counter()
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        try:
            client = self._get_client()
            result = client.detect(
                image_bytes=image_bytes,
                image_id="unknown",
                conf=self.conf
            )
            
        except Exception as e:
            return {
                "cars": 0,
                "pedestrians": 0,
                "bicycles": 0,
                "raw_description": f"gRPC Error: {str(e)}",
                "pipeline_name": self.name,
                "metadata": {"error": str(e)}
            }
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse gRPC response to IR
        ir = parse_yolo_detections_from_mcp(result, pipeline_name=self.name)
        ir['metadata']['latency_ms'] = self._latency_ms
        ir['metadata']['grpc_server'] = f"{self.host}:{self.port}"
        
        return ir
