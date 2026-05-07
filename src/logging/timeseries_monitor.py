"""
Time-series resource monitoring for vision pipeline benchmarking
Samples CPU, GPU, Memory, and Power at regular intervals during processing
"""
import threading
import time
import psutil
from typing import List, Dict, Any, Optional


class TimeSeriesMonitor:
    """
    Background thread that samples system resources at intervals.
    Stops when processing completes.
    """
    
    def __init__(self, sample_interval_ms: float = 100):
        """
        Args:
            sample_interval_ms: Sampling interval in milliseconds (default: 100ms)
        """
        self.sample_interval = sample_interval_ms / 1000.0  # Convert to seconds
        self.samples: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self._monitor_gpu = False
        self._power_monitor = None
        
        # Check if we're on Mac (for GPU/power monitoring via powermetrics)
        import platform
        self.is_mac = platform.system() == "Darwin"
        
        # Initialize power monitor if on Mac
        if self.is_mac:
            try:
                from src.logging.power_monitor import PowerMonitor
                self._power_monitor = PowerMonitor()
            except:
                self._power_monitor = None
    
    def start(self):
        """Start sampling in background thread"""
        self.samples = []
        self.start_time = time.perf_counter()
        self._stop_event.clear()
        
        # Set baseline for process CPU measurement
        self._process = psutil.Process()
        self._process.cpu_percent()  # First call sets baseline, returns 0.0
        
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop sampling"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _sample_loop(self):
        """Background sampling loop"""
        while not self._stop_event.is_set():
            sample = self._take_sample()
            self.samples.append(sample)
            time.sleep(self.sample_interval)
    
    def _take_sample(self) -> Dict[str, Any]:
        """Take a single sample of system resources"""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        sample = {
            "elapsed_ms": elapsed_ms,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": self._process.memory_info().rss / 1024 / 1024,
            "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
        }
        
        # Per-core CPU usage
        sample["cpu_per_core"] = psutil.cpu_percent(percpu=True)
        
        # Process-specific info (now returns actual CPU % since baseline was set)
        sample["process_cpu_percent"] = self._process.cpu_percent()
        sample["process_threads"] = self._process.num_threads()
        
        # Power measurements (if available)
        if self._power_monitor:
            try:
                power_data = self._power_monitor.get_power_sample()
                sample["power"] = power_data
            except:
                pass
        
        return sample
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all collected samples"""
        return self.samples.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from collected samples"""
        if not self.samples:
            return {}
        
        cpu_values = [s["cpu_percent"] for s in self.samples]
        memory_values = [s["memory_percent"] for s in self.samples]
        
        summary = {
            "sample_count": len(self.samples),
            "duration_ms": self.samples[-1]["elapsed_ms"] if self.samples else 0,
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "memory_avg": sum(memory_values) / len(memory_values),
            "memory_max": max(memory_values),
            "memory_min": min(memory_values),
        }
        
        # Process CPU summary
        process_cpu_values = [s.get("process_cpu_percent", 0) for s in self.samples]
        if any(v > 0 for v in process_cpu_values):
            summary["process_cpu_avg"] = sum(process_cpu_values) / len(process_cpu_values)
            summary["process_cpu_max"] = max(process_cpu_values)
            summary["process_cpu_min"] = min(v for v in process_cpu_values if v > 0) if any(v > 0 for v in process_cpu_values) else 0
        
        # Per-core summary
        if self.samples and "cpu_per_core" in self.samples[0]:
            num_cores = len(self.samples[0]["cpu_per_core"])
            core_averages = []
            for i in range(num_cores):
                core_values = [s["cpu_per_core"][i] for s in self.samples if i < len(s["cpu_per_core"])]
                if core_values:
                    core_averages.append(sum(core_values) / len(core_values))
            summary["cpu_per_core_avg"] = core_averages
        
        return summary
    
    def save_to_file(self, filepath: str):
        """Save samples to JSON file for later analysis"""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "sample_interval_ms": self.sample_interval * 1000,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "summary": self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
