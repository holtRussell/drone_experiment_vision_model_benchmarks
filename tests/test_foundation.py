"""
Test the foundation modules
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, check_gpu_available
from src.data import VisDroneLoader, VisDroneGroundTruth
from src.controller import LLMClient
from src.representation import create_ir, validate_ir
from src.prompts import PromptTemplates
from src.logging import TimingStats, ResourceMonitor
from src.cache import CacheManager


def test_config_loading():
    """Test configuration loading"""
    print("Testing config loading...")
    config = load_config('configs/model.yaml')
    assert 'model' in config
    assert 'vllm' in config
    print("  PASSED")


def test_gpu_check():
    """Test GPU availability check"""
    print("Testing GPU check...")
    gpu_info = check_gpu_available()
    assert 'available' in gpu_info
    assert 'type' in gpu_info
    print(f"  GPU: {gpu_info['details']}")
    print("  PASSED")


def test_ir_schema():
    """Test Intermediate Representation schema"""
    print("Testing IR schema...")
    ir = create_ir(cars=5, pedestrians=3, bicycles=2, raw_description="Test scene")
    assert ir['cars'] == 5
    assert ir['pedestrians'] == 3
    assert ir['bicycles'] == 2
    assert validate_ir(ir)
    
    # Test with missing values
    ir_null = create_ir(cars=None, pedestrians=None)
    assert ir_null['cars'] == 0
    assert ir_null['pedestrians'] == 0
    print("  PASSED")


def test_prompts():
    """Test prompt templates"""
    print("Testing prompts...")
    prompts = PromptTemplates('configs/prompts.yaml')
    assert prompts.get_atomic_query('cars') != ""
    assert prompts.get_composite_query() != ""
    print("  PASSED")


def test_timing():
    """Test timing utilities"""
    print("Testing timing...")
    timing = TimingStats()
    timing.start('test_stage')
    import time
    time.sleep(0.01)
    elapsed = timing.stop('test_stage')
    assert elapsed > 0
    print("  PASSED")


def test_resource_monitor():
    """Test resource monitoring"""
    print("Testing resource monitor...")
    monitor = ResourceMonitor(enable_gpu_check=False)
    snapshot = monitor.get_snapshot()
    assert 'cpu_percent' in snapshot
    assert 'memory_used_mb' in snapshot
    print("  PASSED")


def test_cache():
    """Test cache manager"""
    print("Testing cache...")
    cache = CacheManager('test_cache', enabled=True)
    
    # Test vision cache
    test_output = {'cars': 1, 'pedestrians': 2, 'bicycles': 0}
    cache.set_vision_cache('img1', 'yolo_local', test_output)
    cached, hit = cache.get_vision_cache('img1', 'yolo_local')
    assert hit == True
    assert cached['cars'] == 1
    
    # Cleanup
    import shutil
    if os.path.exists('test_cache'):
        shutil.rmtree('test_cache')
    print("  PASSED")


if __name__ == '__main__':
    print("=" * 50)
    print("Foundation Module Tests")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_gpu_check,
        test_ir_schema,
        test_prompts,
        test_timing,
        test_resource_monitor,
        test_cache
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
