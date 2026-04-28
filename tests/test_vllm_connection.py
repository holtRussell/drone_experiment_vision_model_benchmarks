"""
Test vLLM server connection
Run this after starting the vLLM server
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller.llm_client import LLMClient, ModelType
from PIL import Image
import io


def test_vllm_connection():
    """Test basic connection to vLLM server"""
    print("=" * 50)
    print("Testing vLLM Server Connection")
    print("=" * 50)
    
    client = LLMClient()
    
    print(f"\nServer URL: {client.vllm_url}")
    print(f"Model: {client.model_name}")
    print(f"API Key: {client.api_key[:10]}...")
    
    # Test 1: List models
    print("\n--- Test 1: List Models ---")
    try:
        models = client.client.models.list()
        print(f"Available models: {[m.id for m in models.data]}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 2: Text completion
    print("\n--- Test 2: Text Completion ---")
    try:
        response = client.request_text("Say 'Hello from Gemma' in 5 words or less.")
        print(f"Response: {response}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 3: Multimodal (with dummy image)
    print("\n--- Test 3: Multimodal Request ---")
    try:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        
        response, tool_calls, finish_reason = client.request_multimodal(
            prompt="Describe this image briefly.",
            image=img
        )
        print(f"Response: {response[:100]}...")
        print(f"Finish reason: {finish_reason}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("All vLLM connection tests PASSED!")
    print("=" * 50)
    return True


if __name__ == '__main__':
    # Check if server is likely running
    import socket
    
    url = os.environ.get('VLLM_URL', 'http://localhost:8000/v1')
    host = 'localhost'
    port = 8000
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    
    if result != 0:
        print(f"WARNING: Nothing is listening on {host}:{port}")
        print(f"Please start the vLLM server first:")
        print(f"  python -m vllm_mlx.entrypoints.openai.api_server \\")
        print(f"      --model mlx-community/gemma-4-e4b-it-4bit \\")
        print(f"      --port {port}")
        print()
        sys.exit(1)
    
    success = test_vllm_connection()
    sys.exit(0 if success else 1)
