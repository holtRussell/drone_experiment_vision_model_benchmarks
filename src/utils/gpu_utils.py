import platform
from typing import Dict, Optional


def check_gpu_available() -> Dict[str, Optional[str]]:
    """
    Check GPU availability.
    Returns dict with gpu_available (bool) and gpu_type (str or None).
    """
    result = {
        "available": False,
        "type": None,
        "details": None
    }
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        result["available"] = False
        result["type"] = "none"
        result["details"] = "macOS detected - CUDA not available. Using CPU/Metal."
        return result
    
    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["type"] = "cuda"
            result["details"] = f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"
            return result
    except ImportError:
        pass
    
    try:
        import mlx.core as mx
        result["available"] = True
        result["type"] = "mlx"
        result["details"] = "MLX available (Apple Silicon)"
        return result
    except ImportError:
        pass
    
    result["details"] = "No GPU framework detected. Using CPU."
    return result


def get_gpu_memory_usage() -> Dict[str, Optional[float]]:
    """Get GPU memory usage if available"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "total_mb": torch.cuda.get_device_properties(0).total_mem / 1024**2
            }
    except:
        pass
    
    return {"allocated_mb": None, "reserved_mb": None, "total_mb": None}
