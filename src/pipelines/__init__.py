from .base import BasePipeline
from .yolo_local import YoloLocalPipeline
from .vlm_direct import VlmDirectPipeline
from .vlm_multiagent import VlmMultiAgentPipeline

# YoloMcpPipeline - placeholder for custom MCP server integration
try:
    from .yolo_mcp import YoloMcpPipeline
    YOLO_MCP_AVAILABLE = True
except ImportError:
    YoloMcpPipeline = None
    YOLO_MCP_AVAILABLE = False

__all__ = ['BasePipeline', 'YoloLocalPipeline', 'YoloMcpPipeline', 'VlmDirectPipeline', 'VlmMultiAgentPipeline', 'YOLO_MCP_AVAILABLE']
