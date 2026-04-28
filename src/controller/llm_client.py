"""
LLM Client for vLLM - Based on TypeFly's llm_wrapper.py
"""
import os
import io
import base64
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from openai import OpenAI

from src.utils.config import load_config


class ModelType:
    """Model types - using Gemma4:e4b as primary"""
    GEMMA4 = "mlx-community/gemma-4-e4b-it-4bit"


class LLMClient:
    """
    A wrapper for vLLM API using OpenAI client format.
    Reference: TypeFly/typefly/llm_wrapper.py
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with config or environment variables"""
        if config_path:
            config = load_config(config_path)
            self.vllm_url = config.get('vllm', {}).get('url', os.environ.get('VLLM_URL', 'http://localhost:8000/v1'))
            self.api_key = config.get('vllm', {}).get('api_key', os.environ.get('VLLM_API_KEY', 'token-abc123'))
            self.model_name = config.get('model', {}).get('type', ModelType.GEMMA4)
            self.temperature = config.get('model', {}).get('temperature', 0.1)
            self.max_tokens = config.get('model', {}).get('max_tokens', 2048)
        else:
            self.vllm_url = os.environ.get('VLLM_URL', 'http://localhost:8000/v1')
            self.api_key = os.environ.get('VLLM_API_KEY', 'token-abc123')
            self.model_name = ModelType.GEMMA4
            self.temperature = 0.1
            self.max_tokens = 2048
        
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client"""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.vllm_url,
                api_key=self.api_key
            )
        return self._client
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def request_multimodal(
        self,
        prompt: str,
        image: Image.Image,
        model_type: Optional[str] = None,
        tools: Optional[List] = None
    ) -> Tuple[str, Optional[List], str]:
        """
        Send multimodal request with image.
        Returns: (content, tool_calls, finish_reason)
        """
        model = model_type or self.model_name
        image_base64 = self._encode_image(image)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }]
        
        extra_kwargs = {}
        if tools:
            extra_kwargs["tools"] = tools
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra_kwargs
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        
        finish_reason = response.choices[0].finish_reason
        
        return content, tool_calls, finish_reason
    
    def request_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> str:
        """Send text-only request (for processing intermediate representation)"""
        model = model_type or self.model_name
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content or ""
