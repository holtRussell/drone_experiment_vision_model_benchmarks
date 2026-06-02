"""
VLM Multi-Agent Pipeline - Uses VLM in a multi-step agent loop
"""
import time
from typing import Dict, Any
from PIL import Image
from src.pipelines.base import BasePipeline
from src.representation.vlm_parser import parse_vlm_response
from src.controller.llm_client import LLMClient
from src.prompts.templates import PromptTemplates


class VlmMultiAgentPipeline(BasePipeline):
    """Multi-agent VLM processing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("vlm_multiagent", config)
        self.llm_client = None
        self.prompts = None
        self.max_iterations = (config or {}).get('max_iterations', 3)
    
    def _init_client(self):
        if self.llm_client is None:
            self.llm_client = LLMClient()
            self.prompts = PromptTemplates()
    
    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process image with multi-agent VLM approach.
        Uses iterative refinement to improve accuracy.
        """
        self._init_client()
        
        start_time = time.perf_counter()
        
        # Step 1: Initial scene analysis
        initial_query = """Analyze this image and provide:
1. Count of cars (vehicles)
2. Count of pedestrians (people)
3. Count of bicycles
4. Brief scene description

Be precise with counts.

Example:
1. Cars: 7
2. Pedestrians: 15
3. Bicycles: 8
4. This scene contains a crowd of people walking across a busy street with many cars and bikes.

DO NOT use generalizeations such as "dozens" or "many". Instead use specific words to measure each response."""
        
        response1, _, _ = self.llm_client.request_multimodal(initial_query, image)
        
        # Step 2: Refinement query
        refinement_query = f"""Based on the previous analysis: "{response1}"

Please verify and provide final counts and respond in JSON format.

Example: 
Cars: 7,
Pedestrians: 15,
Bicycles: 8,
"""
        
        response2, _, _ = self.llm_client.request_multimodal(refinement_query, image)
        
        # Use the refined response
        self._latency_ms = (time.perf_counter() - start_time) * 1000
        
        ir = parse_vlm_response(response2, pipeline_name=self.name)
        ir['metadata']['latency_ms'] = self._latency_ms
        ir['metadata']['initial_response'] = response1
        ir['metadata']['refined_response'] = response2
        ir['metadata']['iterations'] = self.max_iterations
        
        return ir
