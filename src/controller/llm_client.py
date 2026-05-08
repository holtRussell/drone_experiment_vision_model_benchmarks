"""
LLM Client for Ollama - OpenAI-compatible API
"""
import os
import io
import base64
import requests
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from openai import OpenAI

from src.utils.config import load_config


class ModelType:
    """Model types - using Gemma4:e4b as primary"""
    GEMMA4 = "gemma4:e4b"


class LLMClient:
    """
    A wrapper for Ollama API using OpenAI client format.
    Ollama provides OpenAI-compatible endpoints at http://localhost:11434/v1
    """
    
    def __init__(self, config_path: Optional[str] = None, mock_mode: bool = False):
        """Initialize with config or environment variables"""
        self.mock_mode = mock_mode or os.environ.get('MOCK_LLM', '').lower() == 'true'
        
        if mock_mode:
            self._setup_mock()
        elif config_path:
            config = load_config(config_path)
            self.ollama_url = config.get('ollama', {}).get('url', os.environ.get('OLLAMA_URL', 'http://localhost:11434/v1'))
            self.api_key = config.get('ollama', {}).get('api_key', os.environ.get('OLLAMA_API_KEY', 'ollama'))
            self.model_name = config.get('model', {}).get('type', ModelType.GEMMA4)
            self.temperature = config.get('model', {}).get('temperature', 0.1)
            self.max_tokens = config.get('model', {}).get('max_tokens', 2048)
        else:
            self.ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434/v1')
            self.api_key = os.environ.get('OLLAMA_API_KEY', 'ollama')
            self.model_name = ModelType.GEMMA4
            self.temperature = 0.1
            self.max_tokens = 2048
        
        self._client = None
        self._call_count = 0
        self.last_token_usage = None  # Track token usage per request
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Ollama's local /api/tokenize endpoint.
        This is a local call - no cloud API, no GPU inference.
        """
        if self.mock_mode:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
        
        try:
            response = requests.post(
                "http://localhost:11434/api/tokenize",
                json={"model": self.model_name, "prompt": text},
                timeout=5
            )
            if response.ok:
                tokens = response.json().get("tokens", [])
                return len(tokens)
        except Exception as e:
            print(f"Token counting failed: {e}")
        
        # Fallback estimate
        return len(text) // 4
    
    def count_tokens_multi(self, messages: list) -> int:
        """Count tokens for a list of chat messages"""
        total = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total += self.count_tokens(msg["content"])
            elif isinstance(msg.get("content"), list):
                # Multimodal content
                for item in msg["content"]:
                    if item.get("type") == "text":
                        total += self.count_tokens(item["text"])
                    # Note: Image tokens are harder to count via /api/tokenize
        return total
    
    def _setup_mock(self):
        """Setup mock mode"""
        self.model_name = "mock-gemma-4"
        self.temperature = 0.1
        self.max_tokens = 2048
        print("⚠️  LLM Client running in MOCK mode")
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client (Ollama-compatible)"""
        if self.mock_mode:
            return self
        if self._client is None:
            self._client = OpenAI(
                base_url=self.ollama_url,
                api_key=self.api_key
            )
        return self._client
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response based on prompt"""
        self._call_count += 1
        prompt_lower = prompt.lower()
        
        if "cars" in prompt_lower:
            return "There are approximately 5 cars in the scene."
        elif "pedestrians" in prompt_lower or "people" in prompt_lower:
            return "There are approximately 3 pedestrians in the scene."
        elif "bicycles" in prompt_lower or "bikes" in prompt_lower:
            return "There are approximately 2 bicycles in the scene."
        elif "scene" in prompt_lower or "describe" in prompt_lower:
            return "The scene shows an urban environment with vehicles and people. Multiple cars are visible along with some pedestrians."
        else:
            return f"Mock LLM response #{self._call_count}"
    
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
        if self.mock_mode:
            return self._mock_response(prompt), None, "stop"
        
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
        
        try:
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
        except Exception as e:
            print(f"\nLLM Error: {e}")
            print(f"Falling back to mock mode for remaining requests...")
            self.mock_mode = True
            return self._mock_response(prompt), None, "stop"
    
    def request_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> str:
        """Send text-only request (for processing intermediate representation)"""
        if self.mock_mode:
            # Mock token usage
            self.last_token_usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": 10,
                "total_tokens": len(prompt) // 4 + 10
            }
            return self._mock_response(prompt)
        
        model = model_type or self.model_name
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Count input tokens
        input_tokens = self.count_tokens_multi(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content or ""
            
            # Count output tokens
            output_tokens = self.count_tokens(content)
            
            self.last_token_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            return content
        except Exception as e:
            print(f"\nLLM Error: {e}")
            print(f"Falling back to mock mode for remaining requests...")
            self.mock_mode = True
            return self._mock_response(prompt)
