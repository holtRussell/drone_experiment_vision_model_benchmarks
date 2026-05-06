"""
Quick test script to verify YOLO pipeline works without LLM server
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import YoloLocalPipeline
from src.representation import create_ir, validate_ir
from PIL import Image


def test_yolo_pipeline():
    """Test YOLO pipeline with a sample image"""
    print("Testing YOLO Local Pipeline...")
    
    # Initialize pipeline
    pipeline = YoloLocalPipeline(config={'model': 'yolov8n.pt'})
    
    # Load a test image
    import glob
    image_files = glob.glob("data/visdrone/images/*.jpg")
    
    if not image_files:
        print("No images found in data/visdrone/images/")
        return
    
    test_image_path = image_files[0]
    print(f"Using test image: {test_image_path}")
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Process with YOLO
    print("\nRunning YOLO inference...")
    ir = pipeline.process(image)
    
    print(f"\nResults:")
    print(f"  Cars: {ir['cars']}")
    print(f"  Pedestrians: {ir['pedestrians']}")
    print(f"  Bicycles: {ir['bicycles']}")
    print(f"  Raw description: {ir['raw_description']}")
    print(f"  Model type: {ir['metadata']['model_type']}")
    print(f"  Total detections: {ir['metadata']['total_detections']}")
    
    # Validate IR
    is_valid = validate_ir(ir)
    print(f"\nIR Valid: {is_valid}")
    
    print("\n✓ YOLO pipeline test completed successfully!")


if __name__ == "__main__":
    test_yolo_pipeline()
