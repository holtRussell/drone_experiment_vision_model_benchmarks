"""
YOLO gRPC Service - Based on TypeFly/typefly/serving/yolo_service.py
Provides YOLO object detection via gRPC for the yolo_mcp pipeline
"""
import sys
import os
import gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import time
import grpc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import proto files - add parent to path
from pathlib import Path
proto_dir = Path(__file__).parent / "proto"
sys.path.insert(0, str(proto_dir))

try:
    import hyrch_serving_pb2
    import hyrch_serving_pb2_grpc
except ImportError:
    hyrch_serving_pb2 = None
    hyrch_serving_pb2_grpc = None

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/")
MODEL_TYPE = "yolov8m.pt"


def load_model():
    """Load YOLO model with platform-specific optimizations"""
    if not YOLO_AVAILABLE:
        raise RuntimeError("Ultralytics not installed. Install with: pip install ultralytics")
    
    model_path = os.path.join(MODEL_PATH, MODEL_TYPE)
    
    # Download model if not exists
    if not os.path.exists(model_path):
        os.makedirs(MODEL_PATH, exist_ok=True)
        print(f"Downloading {MODEL_TYPE}...")
        model = YOLO(MODEL_TYPE)  # Will auto-download
    else:
        model = YOLO(model_path)
    
    # Platform-specific device selection
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            model.to('cuda')
            print(f"Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model.to('mps')
            print(f"Using Apple MPS (Metal Performance Shaders)")
        else:
            print("Using CPU")
    
    return model


def release_model(model):
    """Release model and clear memory"""
    del model
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


class YoloService:
    """gRPC service for YOLO object detection"""
    
    def __init__(self, port):
        self.tracking_mode = False
        self.model = load_model()
        self.port = port
    
    def reload_model(self):
        """Reload the YOLO model"""
        if self.model is not None:
            release_model(self.model)
        self.model = load_model()
    
    @staticmethod
    def bytes_to_image(image_bytes) -> Image.Image:
        """Convert bytes to PIL Image"""
        return Image.open(BytesIO(image_bytes))
    
    @staticmethod
    def format_result(yolo_result) -> list:
        """Format YOLO result to JSON-serializable format"""
        if yolo_result.probs is not None:
            print('Warning: Classification task does not support `tojson` yet.')
            return []
        
        formatted_result = []
        data = yolo_result.boxes.data.cpu().tolist()
        h, w = yolo_result.orig_shape
        
        for i, row in enumerate(data):
            # row format: [x1, y1, x2, y2, conf, class_id] or [x1, y1, x2, y2, track_id, conf, class_id]
            if len(row) >= 6:
                if len(row) > 6:  # Tracking mode
                    track_id = int(row[-3])
                    conf = row[-2]
                    class_id = int(row[-1])
                else:
                    track_id = None
                    conf = row[4]
                    class_id = int(row[5])
                
                box = {
                    'x1': round(row[0] / w, 2),
                    'y1': round(row[1] / h, 2),
                    'x2': round(row[2] / w, 2),
                    'y2': round(row[3] / h, 2)
                }
                
                name = yolo_result.names[class_id]
                if track_id is not None:
                    name = f'{name}_{track_id}'
                
                result = {
                    'name': name,
                    'confidence': round(conf, 2),
                    'box': box
                }
                
                # Add segmentation if available
                if yolo_result.masks:
                    x_coords = yolo_result.masks.xy[i][:, 0]
                    y_coords = yolo_result.masks.xy[i][:, 1]
                    result['segments'] = {
                        'x': (x_coords / w).tolist(),
                        'y': (y_coords / h).tolist()
                    }
                
                # Add keypoints if available
                if yolo_result.keypoints is not None:
                    kp_data = yolo_result.keypoints[i].data[0].cpu()
                    x_kp, y_kp, visible = kp_data.unbind(dim=1)
                    result['keypoints'] = {
                        'x': (x_kp / w).tolist(),
                        'y': (y_kp / h).tolist(),
                        'visible': visible.tolist()
                    }
                
                formatted_result.append(result)
        
        return formatted_result
    
    def parse_request(self, request) -> tuple:
        """Parse incoming request"""
        info = json.loads(request.json_data)
        image = YoloService.bytes_to_image(request.image_data)
        
        # Set defaults if missing
        info.setdefault('tracking_mode', False)
        info.setdefault('conf', 0.3)
        
        if self.tracking_mode != info['tracking_mode']:
            self.tracking_mode = info['tracking_mode']
            self.reload_model()
        
        return image, info
    
    def Detect(self, request, context=None):
        """Handle Detect RPC call"""
        try:
            image, info = self.parse_request(request)
            print(f"Received Detect request {info.get('image_id', 'unknown')}")
            
            # Run YOLO inference
            if self.tracking_mode:
                yolo_result = self.model.track(
                    image, 
                    verbose=False, 
                    conf=info['conf'],
                    tracker="bytetrack.yaml"
                )[0]
            else:
                yolo_result = self.model(image, verbose=False, conf=info['conf'])[0]
            
            # Format result
            info['result'] = YoloService.format_result(yolo_result)
            
            # Return response
            if hyrch_serving_pb2 is not None:
                return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))
            else:
                # Fallback if proto not available
                return type('Response', (), {'json_data': json.dumps(info)})()
            
        except Exception as e:
            print(f"Error in Detect: {e}")
            import traceback
            traceback.print_exc()
            if context is not None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            error_response = json.dumps({'error': str(e)})
            if hyrch_serving_pb2 is not None:
                return hyrch_serving_pb2.DetectResponse(json_data=error_response)
            else:
                return type('Response', (), {'json_data': error_response})()


def serve(port, stop_event=None):
    """Start the gRPC server"""
    print(f"Starting YOLO service on port {port}...")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    
    if hyrch_serving_pb2_grpc is not None:
        hyrch_serving_pb2_grpc.add_YoloServiceServicer_to_server(YoloService(port), server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"YOLO service at port {port} [STARTED]")
    
    if stop_event is None:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"YOLO service at port {port} [STOPPED]")
            server.stop(0)
    else:
        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"YOLO service at port {port} [STOPPED]")
            server.stop(0)


if __name__ == "__main__":
    # Test the service
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50050)
    args = parser.parse_args()
    
    serve(args.port)
