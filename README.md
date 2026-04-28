# AI Vision Pipeline Benchmark

Benchmark multiple vision-processing methods on edge devices using the VisDrone dataset.

## Overview

This project evaluates tradeoffs between **latency, resource usage, and scene understanding accuracy** across four pipelines:

1. **YOLO (local inference)** - Direct YOLO object detection
2. **YOLO via MCP server** - YOLO through Model Context Protocol
3. **VLM Direct** - Direct Vision-Language Model processing (Gemma4:e4b)
4. **VLM Multi-Agent** - Multi-agent VLM pipeline

## Architecture

```
Input (Image + Prompt)
        ↓
Vision Processing (always run by default)
        ↓
Intermediate Representation (standardized JSON)
        ↓
Controller LLM (Gemma4:e4b via vLLM)
        ↓
Structured Output (counts + description)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   export VLLM_URL="http://localhost:8000/v1"
   export VLLM_API_KEY="token-abc123"
   ```

3. **Download VisDrone dataset:**
   ```bash
   python src/data/download_visdrone.py train
   ```
   Then organize:
   - Images in `data/visdrone/images/`
   - Annotations in `data/visdrone/annotations/`

4. **Start vLLM server with Gemma4:e4b:**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model mlx-community/gemma-4-e4b-it-4bit \
       --port 8000
   ```

## Usage

### Run tests:
```bash
python tests/test_foundation.py
```

### Run experiment:
```bash
python main.py --config configs/experiment.yaml
```

## Configuration

Edit files in `configs/`:
- `model.yaml` - Model and vLLM settings
- `experiment.yaml` - Dataset, pipelines, queries
- `prompts.yaml` - Prompt templates (version controlled)

## Output

Results stored in `results/`:
- `logs/run_<timestamp>.jsonl` - Structured logs
- `cache/` - Cached vision/LLM outputs
- CSV/JSON exports of metrics

## Requirements

- Python 3.9+
- macOS (tested) or Linux
- vLLM server with Gemma4:e4b model
- YOLO MCP server (for yolo_mcp pipeline)

## Note

GPU monitoring is disabled on macOS. Code checks for GPU availability but runs on CPU by default.
