"""
Timing utilities for granular latency tracking (in milliseconds)
"""
import time
from typing import Dict, Optional, List
from contextlib import contextmanager


class TimingStats:
    """Track timing for various pipeline stages"""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, stage_name: str):
        """Start timing a stage"""
        self.start_times[stage_name] = time.perf_counter()
    
    def stop(self, stage_name: str) -> float:
        """Stop timing a stage and return elapsed ms"""
        if stage_name not in self.start_times:
            return 0.0
        
        elapsed = (time.perf_counter() - self.start_times[stage_name]) * 1000  # ms
        self.timings[stage_name] = elapsed
        del self.start_times[stage_name]
        return elapsed
    
    def get(self, stage_name: str) -> float:
        """Get timing for a stage"""
        return self.timings.get(stage_name, 0.0)
    
    def get_all(self) -> Dict[str, float]:
        """Get all timings"""
        return self.timings.copy()
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging"""
        return self.timings.copy()


@contextmanager
def timed_stage(timing_stats: TimingStats, stage_name: str):
    """Context manager for timing a code block"""
    timing_stats.start(stage_name)
    try:
        yield
    finally:
        timing_stats.stop(stage_name)
