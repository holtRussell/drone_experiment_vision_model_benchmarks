"""
gRPC Client for YOLO Service - Based on TypeFly's service manager
"""
import grpc
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add proto directory to path
proto_dir = Path(__file__).parent / "proto"
import sys
sys.path.insert(0, str(proto_dir))

from hyrch_serving_pb2 import DetectRequest
from hyrch_serving_pb2_grpc import YoloServiceStub


class YoloGrpcClient:
    """Client for YOLO gRPC service"""
    
    def __init__(self, host: str = "localhost", port: int = 50050):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
    
    def connect(self):
        """Create gRPC channel and stub"""
        address = f"{self.host}:{self.port}"
        self.channel = grpc.insecure_channel(address)
        self.stub = YoloServiceStub(self.channel)
    
    def detect(
        self,
        image_bytes: bytes,
        image_id: str = "unknown",
        conf: float = 0.3,
        tracking_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Send detection request to YOLO service.
        
        Args:
            image_bytes: Raw image bytes
            image_id: Identifier for the image
            conf: Confidence threshold
            tracking_mode: Enable tracking mode
            
        Returns:
            Dict with detection results
        """
        if self.stub is None:
            self.connect()
        
        # Prepare request data
        info = {
            "image_id": image_id,
            "conf": conf,
            "tracking_mode": tracking_mode
        }
        
        request = DetectRequest(
            json_data=json.dumps(info),
            image_data=image_bytes
        )
        
        # Call service
        response = self.stub.Detect(request)
        
        # Parse response
        result = json.loads(response.json_data)
        return result
    
    def close(self):
        """Close gRPC channel"""
        if self.channel:
            self.channel.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncYoloGrpcClient:
    """Async gRPC client with channel pooling - Based on TypeFly's ServiceManager"""
    
    def __init__(self):
        self.service_info: Dict[str, tuple[str, list[int]]] = {}
        self.service_channels: Dict[str, asyncio.Queue] = {}
        self.channels_initialized: bool = False
        self.assigned_channels: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.assigned_channels_timeout: int = 10
        self.last_cleanup: float = time.time()
        self.lock = asyncio.Lock()
    
    def add_service(self, service_type: str, host: str, ports: list[int]):
        """Register a service"""
        self.service_info[service_type] = (host, ports)
        self.service_channels[service_type] = asyncio.Queue()
    
    async def initialize_channels(self):
        """Initialize gRPC channels for all services"""
        if self.channels_initialized:
            return
        
        for service_type, (host, ports) in self.service_info.items():
            queue = self.service_channels[service_type]
            for port in ports:
                channel = grpc.aio.insecure_channel(f"{host}:{port}")
                await queue.put(channel)
        
        self.channels_initialized = True
    
    async def get_service_channel(self, service_type: str, robot_info: str):
        """Get a channel for a specific service and robot"""
        async with self.lock:
            await self.initialize_channels()
            await self.clean_dedicated_channels()
            
            if service_type not in self.service_info:
                return "Service not found"
            
            if robot_info not in self.assigned_channels:
                self.assigned_channels[robot_info] = {}
            
            if service_type not in self.assigned_channels[robot_info]:
                try:
                    channel = await self.service_channels[service_type].get_nowait()
                    self.assigned_channels[robot_info][service_type] = {
                        "channel": channel,
                        "timestamp": time.time()
                    }
                    return channel
                except asyncio.QueueEmpty:
                    return f"No available channels for service {service_type}"
            else:
                channel_info = self.assigned_channels[robot_info][service_type]
                channel_info["timestamp"] = time.time()
                return channel_info["channel"]
