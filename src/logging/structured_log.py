"""
Structured logging for experiment runs
"""
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """JSON-based structured logger for experiment runs"""
    
    def __init__(self, log_dir: str = "results/logs", run_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = run_id or str(int(time.time()))
        self.log_file = self.log_dir / f"run_{self.run_id}.jsonl"
        
        self.entries = []
    
    def log(
        self,
        event_type: str,
        data: Dict[str, Any],
        step: Optional[int] = None,
        image_id: Optional[str] = None,
        pipeline: Optional[str] = None
    ):
        """Log a structured event"""
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "run_id": self.run_id,
            "data": data
        }
        
        if step is not None:
            entry["step"] = step
        if image_id:
            entry["image_id"] = image_id
        if pipeline:
            entry["pipeline"] = pipeline
        
        self.entries.append(entry)
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_vision_output(
        self,
        image_id: str,
        pipeline: str,
        ir: Dict[str, Any],
        latency_ms: float,
        cache_hit: bool = False
    ):
        """Log vision processing output"""
        self.log(
            event_type="vision_output",
            data={
                "intermediate_representation": ir,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit
            },
            image_id=image_id,
            pipeline=pipeline
        )
    
    def log_llm_output(
        self,
        image_id: str,
        pipeline: str,
        query_type: str,
        query: str,
        response: str,
        latency_ms: float,
        cache_hit: bool = False
    ):
        """Log LLM response"""
        self.log(
            event_type="llm_output",
            data={
                "query_type": query_type,
                "query": query,
                "response": response,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit
            },
            image_id=image_id,
            pipeline=pipeline
        )
    
    def log_resources(self, resources: Dict[str, Any], image_id: Optional[str] = None):
        """Log resource usage"""
        self.log(
            event_type="resource_usage",
            data=resources,
            image_id=image_id
        )
    
    def save(self):
        """Save all entries to file (already done incrementally)"""
        pass
