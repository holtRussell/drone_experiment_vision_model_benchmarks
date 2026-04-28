"""
Resource monitoring using psutil (CPU, Memory) and optional GPU monitoring
"""
import psutil
from typing import Dict, Optional
from src.utils.gpu_utils import check_gpu_available, get_gpu_memory_usage


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self, enable_gpu_check: bool = True):
        self.enable_gpu_check = enable_gpu_check
        self.gpu_info = check_gpu_available() if enable_gpu_check else {"available": False}
    
    def get_snapshot(self) -> Dict[str, Optional[float]]:
        """Get current resource usage snapshot"""
        snapshot = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_used_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
        }
        
        if self.gpu_info.get("available") and self.enable_gpu_check:
            gpu_mem = get_gpu_memory_usage()
            snapshot.update({
                "gpu_memory_allocated_mb": gpu_mem.get("allocated_mb"),
                "gpu_memory_reserved_mb": gpu_mem.get("reserved_mb"),
                "gpu_memory_total_mb": gpu_mem.get("total_mb"),
                "gpu_type": self.gpu_info.get("type")
            })
        else:
            snapshot.update({
                "gpu_memory_allocated_mb": None,
                "gpu_memory_reserved_mb": None,
                "gpu_memory_total_mb": None,
                "gpu_type": None,
                "gpu_note": self.gpu_info.get("details", "GPU not available")
            })
        
        return snapshot
    
    def get_process_info(self) -> Dict[str, Optional[float]]:
        """Get current process resource usage"""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "fds": process.num_fds() if hasattr(process, 'num_fds') else None
        }
