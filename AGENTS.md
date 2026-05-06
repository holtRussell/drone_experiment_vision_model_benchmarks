# AGENTS.md - OpenCode Agent Guide

## Quick Start

1. **Setup**: `pip install -r requirements.txt` (venv optional but recommended)
2. **Dataset**: Download VisDrone-DET from http://aiskyeye.com/download/ or https://github.com/VisDrone/VisDrone-Dataset
   - Place zips in `data/visdrone/`, extract, organize: `images/*.jpg`, `annotations/*.txt`
3. **vLLM Server**: `vllm-mlx serve mlx-community/gemma-4-e4b-it-4bit --port 8000`
4. **MCP Server** (for yolo_mcp pipeline): `cd image_detection && python object_detection_server.py`
5. **Run**: `python main.py --num-images 10 --pipelines yolo_local yolo_mcp`

## Architecture

```
Input (Image + Prompt)
        ↓
Vision Processing (always run by default, per pipeline)
        ↓
Intermediate Representation (standardized JSON: cars, pedestrians, bicycles)
        ↓
Controller LLM (Gemma4:e4b via vLLM)
        ↓
Structured Output (counts + description)
```

**Key**: Single-pass flow (per user requirement) - NO pre-controller decision step.

## Key Commands

```bash
# Foundation tests (always run first)
python tests/test_foundation.py

# Integration tests
python tests/test_integration.py

# Test vLLM connection (requires running server)
python tests/test_vllm_connection.py

# Run experiment
python main.py --num-images 10 --pipelines yolo_local

# Download VisDrone dataset
python src/data/download_visdrone.py train
```

## Pipeline Configuration

4 pipelines in `src/pipelines/`:
1. `yolo_local` - YOLOv8n (local, Ultralytics) ✅ Ready
2. `yolo_mcp` - Custom MCP server ⏳ Placeholder (waiting for user's code)
3. `vlm_direct` - Direct VLM (Gemma4:e4b) ⚠️ Needs vLLM server
4. `vlm_multiagent` - Multi-agent VLM ⚠️ Needs vLLM server

**Config files**: `configs/model.yaml`, `configs/experiment.yaml`, `configs/prompts.yaml`

## Known Issues

1. **VisDrone download URLs return 404** - Manual download required from aiskyeye.com
2. **yolo_mcp pipeline is placeholder** - User will provide custom MCP server code
3. **grpcio not installed** - gRPC code in `src/serving/` disabled by default
4. **Model name in code vs config** - `yolo_local.py` defaults to `yolov8n.pt`, config says `mshamrai/yolov8n-visdrone`

## Development Notes

- **Python**: 3.9+ (3.13 used in testing)
- **GPU**: No CUDA on Mac (uses MPS/CPU), code checks `check_gpu_available()`
- **Mac-specific**: pynvml disabled, opencv-python-headless used
- **Object categories**: VisDrone has 10 classes, we map to 3 (cars, pedestrians, bicycles)

## Testing Strategy

1. Run `test_foundation.py` (7 tests) - checks configs, IR, caching, timing
2. Run `test_integration.py` (6 tests) - checks imports, flow
3. Test pipelines individually with small image count first
4. Full experiment: 100 images × 4 pipelines × 3 repetitions = 1,200 runs

## File Structure

```
src/
├── controller/llm_client.py  # vLLM client (based on TypeFly)
├── data/                   # VisDrone loading, ground truth
├── pipelines/              # 4 vision pipelines
│   ├── yolo_local.py       # ✅ Ready (YOLOv8)
│   ├── yolo_mcp.py         # ⏳ Placeholder (your MCP code)
│   ├── vlm_direct.py       # Gemma4:e4b direct
│   └── vlm_multiagent.py  # Multi-agent VLM
├── representation/          # IR schema, parsers
├── prompts/               # Prompt templates (version controlled)
├── cache/                 # Vision + LLM caching
├── logging/               # Timing, resources, structured logs
├── experiment/            # Experiment runner
├── evaluation/            # Metrics (MAE, RMSE, Exact Match)
├── analysis/              # Comparison, visualization
└── utils/                 # Config loader, GPU utils
```

## Gotchas

- **YOLO model**: `mshamrai/yolov8n-visdrone` is HuggingFace model (auto-downloads), not local file
- **vLLM package**: Use `vllm-mlx` (not standard `vllm`) on Mac
- **Dataset structure**: Expects `data/visdrone/images/*.jpg` and `data/visdrone/annotations/*.txt`
- **Syntax errors**: Some files had editing errors during development - run `python -m py_compile src/**/*.py` to verify

## User Constraints

- Single-pass architecture (no pre-controller review)
- Gemma4:e4b model for LLM processing
- VisDrone dataset (10 object categories → 3 output categories)
- Mac compatibility (no CUDA, optional GPU checks)
- YOLO MCP server code will be provided later
