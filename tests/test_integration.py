"""
Integration test for the full pipeline
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import VisDroneLoader
from src.data.ground_truth import VisDroneGroundTruth
from src.representation import create_ir, validate_ir
from src.prompts import PromptTemplates
from src.logging import TimingStats, ResourceMonitor, StructuredLogger
from src.cache import CacheManager
from src.controller.llm_client import LLMClient


def test_imports():
    """Test all modules can be imported"""
    print("Testing imports...")
    # imports done above
    print("  PASSED")


def test_config_flow():
    """Test configuration loading flow"""
    print("Testing config flow...")
    from src.utils import load_all_configs
    configs = load_all_configs()
    assert 'model' in configs
    assert 'experiment' in configs
    assert 'prompts' in configs
    print("  PASSED")


def test_ir_flow():
    """Test IR creation and validation"""
    print("Testing IR flow...")
    ir = create_ir(cars=5, pedestrians=3, bicycles=1)
    assert validate_ir(ir)
    assert ir['cars'] == 5
    assert ir['pedestrians'] == 3
    assert ir['bicycles'] == 1
    print("  PASSED")


def test_cache_flow():
    """Test cache set/get flow"""
    print("Testing cache flow...")
    cache = CacheManager('test_integration_cache', enabled=True)
    
    test_ir = create_ir(cars=2, pedestrians=1, bicycles=0)
    cache.set_vision_cache('test_img', 'yolo_local', test_ir)
    
    retrieved, hit = cache.get_vision_cache('test_img', 'yolo_local')
    assert hit == True
    assert retrieved['cars'] == 2
    
    import shutil
    if os.path.exists('test_integration_cache'):
        shutil.rmtree('test_integration_cache')
    
    print("  PASSED")


def test_prompt_flow():
    """Test prompt template loading"""
    print("Testing prompt flow...")
    prompts = PromptTemplates()
    
    # Test IR prompt generation
    ir = create_ir(cars=3, pedestrians=2, bicycles=1)
    ir_prompt = prompts.get_ir_prompt(ir, "How many cars are there?")
    
    assert "Cars: 3" in ir_prompt
    assert "How many cars are there?" in ir_prompt
    print("  PASSED")


def test_llm_client_init():
    """Test LLM client initialization"""
    print("Testing LLM client init...")
    client = LLMClient()
    assert client.vllm_url is not None
    assert client.model_name is not None
    print("  PASSED (client init only, no API call)")


if __name__ == '__main__':
    print("=" * 50)
    print("Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_flow,
        test_ir_flow,
        test_cache_flow,
        test_prompt_flow,
        test_llm_client_init,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed > 0:
        sys.exit(1)
