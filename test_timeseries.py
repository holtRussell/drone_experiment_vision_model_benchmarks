"""
Quick test script to verify time-series monitor and vision metrics
Run: python test_timeseries.py
"""
import time
from src.logging.timeseries_monitor import TimeSeriesMonitor
from src.logging.structured_log import StructuredLogger
from src.representation.schema import create_ir

def test_timeseries_monitor():
    """Test basic time-series monitoring"""
    print("Testing TimeSeriesMonitor...")
    
    monitor = TimeSeriesMonitor(sample_interval_ms=100)
    monitor.start()
    
    print("  Monitoring for 2 seconds...")
    time.sleep(2)
    
    monitor.stop()
    samples = monitor.get_samples()
    summary = monitor.get_summary()
    
    print(f"  Collected {len(samples)} samples")
    print(f"  Duration: {summary.get('duration_ms', 0):.1f}ms")
    print(f"  CPU avg: {summary.get('cpu_avg', 0):.1f}%")
    print(f"  Memory avg: {summary.get('memory_avg', 0):.1f}%")
    
    # Save to file
    monitor.save_to_file("results/test_timeseries.json")
    print("  Saved to results/test_timeseries.json")
    
    return True

def test_vision_metrics_logging():
    """Test vision metrics logging"""
    print("\nTesting vision metrics logging...")
    
    logger = StructuredLogger(log_dir="results/test_logs", run_id="test_vision")
    
    # Create IR with enhanced metrics
    ir = create_ir(
        cars=5,
        pedestrians=3,
        bicycles=2,
        pipeline_name="yolo_local",
        confidence_scores=[0.9, 0.85, 0.88, 0.76, 0.92, 0.81, 0.79, 0.95, 0.83, 0.87],
        avg_confidence_per_class={
            "car": 0.89,
            "pedestrian": 0.82,
            "bicycle": 0.86
        },
        image_resolution=(1920, 1080),
        model_input_size=(640, 640)
    )
    
    # Log vision metrics
    logger.log_vision_metrics(
        image_id="test_001",
        pipeline="yolo_local",
        data={
            "cars": ir.get("cars", 0),
            "pedestrians": ir.get("pedestrians", 0),
            "bicycles": ir.get("bicycles", 0),
            "total_detections": len(ir.get("metadata", {}).get("confidence_scores", [])),
            "confidence_scores": ir.get("metadata", {}).get("confidence_scores", []),
            "avg_confidence": sum(ir.get("metadata", {}).get("confidence_scores", [])) / max(len(ir.get("metadata", {}).get("confidence_scores", [])), 1),
            "avg_confidence_per_class": ir.get("metadata", {}).get("avg_confidence_per_class", {}),
            "image_resolution": ir.get("metadata", {}).get("image_resolution"),
            "model_input_size": ir.get("metadata", {}).get("model_input_size"),
            "latency_ms": 120.5,
            "cache_hit": False
        }
    )
    
    print("  Logged vision metrics to results/test_logs/")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Time-Series Monitor & Vision Metrics Test")
    print("=" * 50)
    
    try:
        test_timeseries_monitor()
        test_vision_metrics_logging()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
