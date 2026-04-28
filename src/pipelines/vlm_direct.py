"""
VLM Direct Pipeline - Sends image directly to VLM (Gemma4:e4b) for processing
"""
import time
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.vlm_parser import parse_vlm_response
from src.controller.llm_client import LLMClient
from src.prompts.templates import PromptTemplates


class VlmDirectPipeline(BasePipeline):
    """Direct VLM processing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("vlm_direct", config)
        self.llm_client = None
        self.prompts = None
    
    def _init_client(self):
        """Lazy-initialize LLM client"""
        if self.llm_client is None:
            self.llm_client = LLMClient()
            self.prompts = PromptTemplates()
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image directly with VLM.
        
        Args:
            image: PIL Image
            
        Returns:
            Intermediate Representation
        """
        self._init_client()
        
        start_time = time.perf_counter()
        
        # Use composite query to get scene description
        composite_query = self.prompts.get_composite_query()
        
        # Call VLM with image
        response, _, _ = self.llm_client.request_multimodal(
            prompt=composite_query,
            image=image
        )
        
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse VLM response to IR
        ir = parse_vlm_response(response, pipeline_name=self.name)
        ir['metadata']['latency_ms'] = self._latency_ms
        ir['metadata']['raw_vlm_response'] = response
        
        return ir
