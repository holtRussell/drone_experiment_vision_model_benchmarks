"""
Cache Manager for vision and LLM outputs
"""
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class CacheManager:
    """Manage caching of vision and LLM outputs"""
    
    def __init__(self, cache_dir: str = "results/cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        
        self.vision_cache_dir = self.cache_dir / "vision"
        self.llm_cache_dir = self.cache_dir / "llm"
        
        self.vision_cache_dir.mkdir(exist_ok=True)
        self.llm_cache_dir.mkdir(exist_ok=True)
        
        self.hits = {"vision": 0, "llm": 0}
        self.misses = {"vision": 0, "llm": 0}
    
    def _hash_key(self, *args) -> str:
        """Create hash key from arguments"""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_vision_cache(
        self,
        image_id: str,
        pipeline: str
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Get cached vision output.
        Returns: (output, cache_hit)
        """
        if not self.enabled:
            return None, False
        
        cache_key = self._hash_key(image_id, pipeline)
        cache_file = self.vision_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.hits["vision"] += 1
                return json.load(f), True
        
        self.misses["vision"] += 1
        return None, False
    
    def set_vision_cache(
        self,
        image_id: str,
        pipeline: str,
        output: Dict[str, Any]
    ):
        """Cache vision output"""
        if not self.enabled:
            return
        
        cache_key = self._hash_key(image_id, pipeline)
        cache_file = self.vision_cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(output, f)
    
    def get_llm_cache(
        self,
        input_text: str
    ) -> Tuple[Optional[str], bool]:
        """
        Get cached LLM output.
        Returns: (output, cache_hit)
        """
        if not self.enabled:
            return None, False
        
        cache_key = self._hash_key(input_text)
        cache_file = self.llm_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.hits["llm"] += 1
                data = json.load(f)
                return data.get("output"), True
        
        self.misses["llm"] += 1
        return None, False
    
    def set_llm_cache(
        self,
        input_text: str,
        output: str
    ):
        """Cache LLM output"""
        if not self.enabled:
            return
        
        cache_key = self._hash_key(input_text)
        cache_file = self.llm_cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({"input": input_text, "output": output}, f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "enabled": self.enabled,
            "hits": self.hits.copy(),
            "misses": self.misses.copy(),
            "total_requests": {
                "vision": self.hits["vision"] + self.misses["vision"],
                "llm": self.hits["llm"] + self.misses["llm"]
            }
        }
