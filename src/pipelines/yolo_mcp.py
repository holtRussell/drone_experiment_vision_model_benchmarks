"""
YOLO via MCP Pipeline - Integrates with YOLO MCP Server
Communicates with the image_detection/object_detection_server.py via MCP protocol
"""
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.pipelines.mcp_client import MCPClient
from src.representation.yolo_parser import parse_yolo_detections_from_mcp


class YoloMcpPipeline(BasePipeline):
    """YOLO inference via MCP Server (image_detection/object_detection_server.py)"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("yolo_mcp", config)
        self.mcp_url = (config or {}).get('mcp_url', 'http://localhost:8099')
        self._client = None

    @property
    def client(self) -> MCPClient:
        """Lazy-load MCP client"""
        if self._client is None:
            self._client = MCPClient(base_url=self.mcp_url)
        return self._client

    def _save_image_temp(self, image: Image.Image) -> str:
        """Save PIL image to temporary file and return path"""
        temp_dir = Path(tempfile.gettempdir()) / "yolo_mcp_images"
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"image_{int(time.time() * 1000)}.jpg"
        image.save(temp_path, "JPEG")
        return str(temp_path)

    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image via MCP Server.

        Args:
            image: PIL Image

        Returns:
            Intermediate Representation
        """
        start_time = time.perf_counter()

        # Save image to temp file (MCP server needs file path)
        temp_image_path = self._save_image_temp(image)

        try:
            # Call MCP server's detect_objects_in_image tool
            mcp_response = self.client.call_tool(
                tool_name="detect_objects_in_image",
                arguments={"image_location": temp_image_path}
            )

            # Parse MCP response to Intermediate Representation
            ir = parse_yolo_detections_from_mcp(mcp_response, pipeline_name=self.name)
            ir['metadata']['mcp_url'] = self.mcp_url
            ir['metadata']['image_url'] = mcp_response.get('image_url')

        except ConnectionError as e:
            raise ConnectionError(
                f"MCP Server connection failed: {e}\n"
                f"Make sure the server is running:\n"
                f"  cd image_detection && python object_detection_server.py"
            )
        except Exception as e:
            raise RuntimeError(f"MCP pipeline error: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        self._latency_ms = (time.perf_counter() - start_time) * 1000
        ir['metadata']['latency_ms'] = self._latency_ms

        return ir
