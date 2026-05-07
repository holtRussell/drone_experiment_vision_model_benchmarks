"""
Power monitoring for Apple Silicon using powermetrics
Requires sudo privileges for accurate power measurements
"""
import subprocess
import re
import platform
import time
from typing import Optional, Dict, Any


class PowerMonitor:
    """
    Monitor power consumption on Apple Silicon Macs.
    Uses powermetrics (requires sudo for real-time measurements).
    """
    
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
        self.powermetrics_available = self._check_powermetrics()
        self.last_sample = None
    
    def _check_powermetrics(self) -> bool:
        """Check if powermetrics is available"""
        if not self.is_mac:
            return False
        try:
            result = subprocess.run(
                ["which", "powermetrics"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def get_power_sample(self) -> Dict[str, Any]:
        """
        Get single power sample using powermetrics.
        Note: This requires sudo privileges for accurate measurements.
        
        Returns:
            Dict with power measurements or estimates
        """
        if not self.powermetrics_available:
            return self._estimate_power()
        
        try:
            # Run powermetrics for 1 sample with 1 second interval
            # Using plist format for easier parsing
            result = subprocess.run(
                ["sudo", "powermetrics", 
                 "--samplers", "cpu_power,gpu_power",
                 "-i", "1000",  # 1 second interval
                 "-n", "1",     # 1 sample
                 "--format", "plist"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return self._parse_plist_output(result.stdout)
            else:
                # Fall back to estimation if sudo fails
                return self._estimate_power()
            
        except subprocess.TimeoutExpired:
            return {"error": "powermetrics timeout", **self._estimate_power()}
        except Exception as e:
            return {"error": str(e), **self._estimate_power()}
    
    def _parse_plist_output(self, output: bytes) -> Dict[str, Any]:
        """Parse plist output from powermetrics"""
        try:
            import plistlib
            data = plistlib.loads(output)
            
            power_data = {
                "source": "powermetrics",
                "timestamp": time.time(),
            }
            
            # Extract CPU power
            if "CPUPower" in data:
                cpu_data = data["CPUPower"]
                # Sum all CPU core power
                cpu_power = 0
                for core in cpu_data.get("cores", []):
                    cpu_power += core.get("power", 0)
                power_data["cpu_power_w"] = cpu_power
            
            # Extract GPU power
            if "GPUPower" in data:
                gpu_data = data["GPUPower"]
                power_data["gpu_power_w"] = gpu_data.get("power", 0)
            
            # Calculate total
            cpu = power_data.get("cpu_power_w", 0)
            gpu = power_data.get("gpu_power_w", 0)
            if cpu or gpu:
                power_data["total_power_w"] = cpu + gpu
            
            # Get GPU utilization if available
            if "GPUActivity" in data:
                gpu_activity = data["GPUActivity"]
                power_data["gpu_utilization_percent"] = gpu_activity.get("accelerator_busy", 0)
            
            return power_data
            
        except Exception as e:
            # Fall back to text parsing
            return self._parse_text_output(output.decode('utf-8', errors='ignore'))
    
    def _parse_text_output(self, output: str) -> Dict[str, Any]:
        """Fallback: Parse text output from powermetrics"""
        power_data = {
            "source": "powermetrics_text",
            "timestamp": time.time(),
        }
        
        # Parse CPU power
        cpu_match = re.search(r'CPU Power:\s*([\d.]+)\s*W', output)
        if cpu_match:
            power_data["cpu_power_w"] = float(cpu_match.group(1))
        
        # Parse GPU power  
        gpu_match = re.search(r'GPU Power:\s*([\d.]+)\s*W', output)
        if gpu_match:
            power_data["gpu_power_w"] = float(gpu_match.group(1))
        
        # Calculate total
        cpu = power_data.get("cpu_power_w", 0)
        gpu = power_data.get("gpu_power_w", 0)
        if cpu or gpu:
            power_data["total_power_w"] = cpu + gpu
        
        return power_data
    
    def _estimate_power(self) -> Dict[str, Any]:
        """
        Estimate power based on CPU/GPU utilization.
        Less accurate but works without sudo.
        """
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Rough estimates for Apple Silicon M-series
        # These are approximations - real measurement needs powermetrics
        cpu_tdp_estimate = 15  # Watts (typical M1/M2/M3)
        estimated_cpu_w = (cpu_percent / 100) * cpu_tdp_estimate
        
        return {
            "cpu_power_w": estimated_cpu_w,
            "gpu_power_w": None,  # Can't estimate without GPU utilization
            "total_power_w": estimated_cpu_w,
            "source": "estimation",
            "note": "Estimated from CPU utilization. Use sudo for accurate powermetrics.",
            "cpu_utilization_percent": cpu_percent,
            "timestamp": time.time(),
        }
    
    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage (requires powermetrics)"""
        if not self.powermetrics_available:
            return None
        
        try:
            result = subprocess.run(
                ["sudo", "powermetrics",
                 "--samplers", "gpu",
                 "-i", "1000",
                 "-n", "1",
                 "--format", "text"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse GPU utilization
                match = re.search(r'GPU\s*:\s*(\d+)%', result.stdout)
                if match:
                    return float(match.group(1))
        except:
            pass
        
        return None
