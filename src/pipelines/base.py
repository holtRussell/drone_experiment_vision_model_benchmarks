"""
Base Pipeline Interface
All pipelines must implement the process() method.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image


class BasePipeline(ABC):
    """Abstract base class for vision processing pipelines"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._latency_ms = 0
    
    @abstractmethod
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image and return Intermediate Representation.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation dict with keys:
            - cars: int
            - pedestrians: int
            - bicycles: int
            - raw_description: str (optional)
        """
        pass
    
    @property
    def latency_ms(self) -> int:
        """Get last processing latency in milliseconds"""
        return self._latency_ms
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline info for logging"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config
        }
