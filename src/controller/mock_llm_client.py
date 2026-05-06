"""
Mock LLM Client for testing without vLLM server
"""
from typing import Optional, List, Dict, Any, Tuple


class MockLLMClient:
    """Mock LLM client that returns predefined responses"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model_name = "mock-gemma-4"
        self.temperature = 0.1
        self.max_tokens = 2048
        self._call_count = 0
    
    @property
    def client(self):
        return self
    
    def request_multimodal(
        self,
        prompt: str,
        image: Any,
        model_type: Optional[str] = None,
        tools: Optional[List] = None
    ) -> Tuple[str, Optional[List], str]:
        return f"Mock response to: {prompt[:50]}...", None, "stop"
    
    def request_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> str:
        self._call_count += 1
        
        # Generate mock response based on prompt content
        if "cars" in prompt.lower():
            return "There are approximately 5 cars in the scene."
        elif "pedestrians" in prompt.lower():
            return "There are approximately 3 pedestrians in the scene."
        elif "bicycles" in prompt.lower():
            return "There are approximately 2 bicycles in the scene."
        elif "scene" in prompt.lower() or "describe" in prompt.lower():
            return "The scene shows an urban environment with vehicles and people. Cars are parked along the street. Several pedestrians are walking on the sidewalk."
        else:
            return f"Mock LLM response #{self._call_count} to query."
