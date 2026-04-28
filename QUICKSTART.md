# QuickStart Guide - Vision Pipeline Benchmark

A complete guide to get the benchmark running on your system.

## Prerequisites

- **Python**: 3.9+ (3.13 recommended based on your TypeFly setup)
- **System**: macOS (Apple Silicon M-series) or Linux
- **Model**: Gemma4:e4b (mlx-community/gemma-4-e4b-it-4bit)
- **Dataset**: VisDrone (will be downloaded automatically)

---

## Step 1: Install Dependencies

```bash
cd /Users/holtrussell/Desktop/Python/drone_experiment_vision_model_benchmarks

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

**Key dependencies installed:**
- `vllm-mlx` - vLLM for Apple Silicon
- `ultralytics` - YOLO models
- `openai` - OpenAI-compatible client for vLLM API
- `psutil` - System resource monitoring

---

## Step 2: Start vLLM Server (Gemma4:e4b)

### For macOS (Apple Silicon) - Using vLLM MLX:

**Option 1: Use the vllm-mlx CLI (Recommended):**
```bash
# Start the vLLM MLX server with Gemma4:e4b
vllm-mlx serve mlx-community/gemma-4-e4b-it-4bit --port 8000
```

**Option 2: Use Python module directly:**
```bash
python -m vllm_mlx.cli serve --model mlx-community/gemma-4-e4b-it-4bit --port 8000
```

The server should output something like:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Troubleshooting:**
- If `vllm-mlx` command not found, ensure venv is activated: `source venv/bin/activate`
- If module not found, install with: `pip install vllm-mlx`
- The package name is `vllm-mlx` (not `vllm`)

### Verify Server is Running:

Open a new terminal and test:
```bash
curl http://localhost:8000/v1/models
```

Expected response:
```json
{"data": [{"id": "mlx-community/gemma-4-e4b-it-4bit", ...}], "object": "list"}
```

---

## Step 3: Download VisDrone Dataset

```bash
# Download training subset (recommended for initial testing)
python src/data/download_visdrone.py train

# Or download all subsets
python src/data/download_visdrone.py all
```

After download, organize the dataset:
```bash
cd data/visdrone

# Create directory structure
mkdir -p images annotations

# Move images to images/
# Move annotation txt files to annotations/
# (The download script may need manual organization)
```

**Dataset structure should be:**
```
data/visdrone/
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
└── annotations/
    ├── 00001.txt
    ├── 00002.txt
    └── ...
```

---

## Step 4: Configure Environment Variables

Create a `.env` file (optional):
```bash
# .env file
VLLM_URL=http://localhost:8000/v1
VLLM_API_KEY=token-abc123
```

Or export directly:
```bash
export VLLM_URL="http://localhost:8000/v1"
export VLLM_API_KEY="token-abc123"
```

---

## Step 5: Run Tests

```bash
# Test foundation modules
python tests/test_foundation.py

# Test integration
python tests/test_integration.py
```

Expected output:
```
==================================================
Foundation Module Tests
==================================================
Testing config loading...
  PASSED
...
Results: 7 passed, 0 failed
==================================================
```

---

## Step 6: Run Benchmark

### Quick Test (10 images, 1 pipeline):
```bash
python main.py --num-images 10 --pipelines yolo_local
```

### Full Benchmark (100 images, all pipelines):
```bash
python main.py \
    --config configs/experiment.yaml \
    --num-images 100 \
    --pipelines yolo_local yolo_mcp vlm_direct vlm_multiagent
```

### Run Specific Pipeline:
```bash
# Only YOLO local
python main.py --pipelines yolo_local --num-images 50

# Only VLM direct
python main.py --pipelines vlm_direct --num-images 50
```

---

## Step 7: View Results

Results are saved in `results/`:
```bash
# View logs
ls results/logs/
cat results/logs/run_*.jsonl

# View cache statistics
ls results/cache/vision/
ls results/cache/llm/
```

---

## Configuration Files

Edit these files to customize behavior:

- **`configs/model.yaml`** - Model settings, vLLM endpoint
- **`configs/experiment.yaml`** - Dataset path, image count, pipelines
- **`configs/prompts.yaml`** - Prompt templates (version controlled)

---

## Troubleshooting

### Issue: "vllm_mlx not found"
**Solution:** Install vLLM MLX:
```bash
pip install vllm-mlx
```

### Issue: "Cannot connect to vLLM server"
**Solution:** 
1. Check if server is running: `curl http://localhost:8000/v1/models`
2. Verify port 8000 is not in use: `lsof -i :8000`
3. Check `VLLM_URL` environment variable

### Issue: "VisDrone dataset not found"
**Solution:**
1. Run download script: `python src/data/download_visdrone.py train`
2. Verify `data/visdrone/images/` exists
3. Update `configs/experiment.yaml` with correct `dataset_path`

### Issue: "No module named 'ultralytics'"
**Solution:**
```bash
pip install ultralytics
```

### Issue: "psutil not found"
**Solution:**
```bash
pip install psutil
```

---

## Next Steps

1. **Provide YOLO MCP Server Code**: Once you share it, I'll integrate it into `src/pipelines/yolo_mcp.py`

2. **Run Full Benchmark**: After integration, run all 4 pipelines

3. **Analyze Results**: Use analysis module to compare pipelines:
   ```python
   from src.analysis import load_results, compare_pipelines
   results = load_results()
   comparison = compare_pipelines(results)
   ```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python tests/test_foundation.py` | Test foundation modules |
| `python tests/test_integration.py` | Test integration |
| `python main.py --num-images 10` | Run quick test |
| `python main.py --help` | Show all options |
| `python src/data/download_visdrone.py train` | Download dataset |

---

## File Structure

```
drone_experiment_vision_model_benchmarks/
├── configs/           # YAML configuration files
├── src/              # Source code
│   ├── data/         # Dataset loading, ground truth
│   ├── controller/   # LLM client (vLLM)
│   ├── pipelines/    # 4 vision pipelines
│   ├── representation/ # Intermediate representation
│   ├── prompts/      # Prompt templates
│   ├── cache/        # Cache management
│   ├── logging/      # Timing, resources, logs
│   ├── experiment/   # Experiment runner
│   ├── evaluation/   # Metrics
│   ├── analysis/      # Comparison, visualization
│   └── utils/        # Config, GPU utils
├── tests/            # Test files
├── data/             # Dataset (gitignored)
├── results/          # Output (gitignored)
├── requirements.txt  # Dependencies
├── QUICKSTART.md     # This file
├── README.md         # Project overview
└── main.py          # Entry point
```

---

**Last Updated**: April 2026  
**Tested On**: macOS (Apple Silicon M-series), Python 3.13
