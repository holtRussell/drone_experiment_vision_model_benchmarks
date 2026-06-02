# Experiment Setup and Requirements

## Hardware

| Component | Specification | Notes |
|---|---|---|
| **CPU** | Apple Silicon (M-series) or x86 Linux | Tested on macOS (M-series). No CUDA required. |
| **RAM** | 8GB+ (16GB+ recommended for VLM pipelines) | VLM pipelines (Gemma4:e4b) are memory-intensive. |
| **GPU** | Optional — Apple Silicon Neural Engine or NVIDIA GPU | GPU monitoring disabled on macOS. Falls back to CPU by default. |
| **Storage** | ~5GB for dataset + models | VisDrone dataset ~1.5GB, YOLOv8m ~50MB, Gemma4:e4b ~4GB. |

## Operating System

- **macOS** (primary development target, tested)
- **Linux** (supported but less tested)

---

## Python Environment

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Runtime (3.13 used in testing) |
| **pip** | Latest | Package management |

---

## Core Packages

### Data Processing & Scientific Computing

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24.0 | Numerical operations (array math, metrics computation) |
| `pandas` | ≥2.0.0 | DataFrame manipulation, result aggregation, summary statistics |
| `pillow` | ≥10.0.0 | Image loading, resizing, RGB conversion for pipeline input |
| `pyyaml` | ≥6.0 | Configuration file parsing (YAML → Python dict) |
| `scipy` | ≥1.11.0 | Scientific utilities (supplementary statistical functions) |

### Vision Processing

| Package | Version | Purpose |
|---|---|---|
| **`ultralytics`** | ≥8.0.0 | YOLOv8 model inference (yolo_local and yolo_mcp pipelines). Loads models from `.pt` files or HuggingFace. |
| **`opencv-python-headless`** | ≥4.8.0 (macOS) | Image processing helper functions (used internally by ultralytics) |
| **`opencv-python`** | ≥4.8.0 (Linux/Windows) | Same, with GUI support for non-macOS |

### LLM / Controller

| Package | Version | Purpose |
|---|---|---|
| **`vllm-mlx`** | ≥0.1.0 | Apple Silicon MLX-optimized vLLM server. Serves Gemma4:e4b via OpenAI-compatible API. **Required for VLM pipelines and controller LLM.** |
| **`openai`** | ≥1.0.0 | OpenAI-compatible client library. Used by `LLMClient` to make requests to the vLLM server (both text and multimodal). |

### Monitoring & System

| Package | Version | Purpose |
|---|---|---|
| **`psutil`** | ≥5.9.0 | System resource monitoring (CPU %, memory %, process-level stats). Used by `ResourceMonitor` and `TimeSeriesMonitor` for per-phase resource tracking. |

### Visualization

| Package | Version | Purpose |
|---|---|---|
| **`matplotlib`** | ≥3.7.0 | Publication-quality chart generation (boxplots, bar charts, scatter plots, Gantt charts). |
| **`seaborn`** | ≥0.12.0 | Statistical data visualization (heatmaps, boxplots with grouped categories). Built on matplotlib. |

### Evaluation

| Package | Version | Purpose |
|---|---|---|
| **`scikit-learn`** | ≥1.3.0 | Metrics utilities (MAE, RMSE — though implemented manually in this project). |

### Utilities

| Package | Version | Purpose |
|---|---|---|
| **`requests`** | ≥2.31.0 | HTTP client for MCP server communication and vLLM API calls (fallback). |
| **`tqdm`** | ≥4.65.0 | Progress bar for experiment runs. |
| **`python-dotenv`** | ≥1.0.0 | Environment variable loading from `.env` files (API keys, URLs). |
| **`argparse`** | ≥1.4.0 | CLI argument parsing (built-in, listed for reference). |

---

## External Services / Servers

| Service | Port | Purpose | How to Start |
|---|---|---|---|
| **vLLM Server** (vllm-mlx) | `8000` | Serves Gemma4:e4b model. Handles both VLM image-to-text (multimodal) and controller LLM text queries. | `vllm-mlx serve mlx-community/gemma-4-e4b-it-4bit --port 8000` |
| **YOLO MCP Server** (optional) | `8099` | Custom object detection server for the yolo_mcp pipeline. Streamable HTTP transport with JSON-RPC. | `cd image_detection && python object_detection_server.py` |

---

## Dataset

| Dataset | Source | Contents | Location |
|---|---|---|---|
| **VisDrone-DET** | [aiskyeye.com](http://aiskyeye.com/download/) or [GitHub](https://github.com/VisDrone/VisDrone-Dataset) | ~10,000 images with 10 object categories (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). | `data/visdrone/images/*.jpg`, `data/visdrone/annotations/*.txt` |

The experiment maps the 10 VisDrone categories down to 3 output categories:
- **Cars**: car, van, truck, bus
- **Pedestrians**: pedestrian, people
- **Bicycles**: bicycle, tricycle, awning-tricycle, motor

---

## Models

| Model | Source | Size | Purpose |
|---|---|---|---|
| **Gemma4:e4b** (4-bit quantized) | HuggingFace via `mlx-community/gemma-4-e4b-it-4bit` | ~4GB | Controller LLM + VLM vision pipelines. Runs on vLLM server. |
| **YOLOv8m** (Medium) | Local `.pt` file (`models/yolov8m.pt`) or Ultralytics auto-download | ~50MB | Object detection for yolo_local and yolo_mcp pipelines. COCO-trained (80 classes). Also configurable as `mshamrai/yolov8n-visdrone` (VisDrone-finetuned). |

---

## Configuration Files

| File | Purpose |
|---|---|
| `configs/experiment.yaml` | Experiment parameters: dataset path, image count, repetitions, pipeline selection, caching, monitoring. |
| `configs/model.yaml` | LLM model config: model name, temperature, max tokens, Ollama endpoint, system prompt. |
| `configs/prompts.yaml` | Prompt templates: atomic queries (per object type), composite query, IR prompt template. |

---

## File Layout

```
drone_experiment_vision_model_benchmarks/
├── main.py                        # Entry point
├── requirements.txt               # Python dependencies
├── configs/                       # YAML configuration files
│   ├── experiment.yaml
│   ├── model.yaml
│   └── prompts.yaml
├── data/visdrone/                 # Dataset (downloaded separately)
│   ├── images/*.jpg
│   └── annotations/*.txt
├── models/                        # Local model files
│   └── yolov8m.pt
├── src/
│   ├── pipelines/                 # 4 vision pipelines
│   ├── controller/                # LLM client (vLLM/Ollama)
│   ├── representation/            # IR schema, parsers
│   ├── evaluation/                # Metrics (MAE, RMSE, Exact Match)
│   ├── experiment/                # Experiment runner
│   ├── analysis/                  # Visualization and comparison
│   ├── cache/                     # Vision + LLM caching
│   ├── logging/                   # Structured logging, timing, monitoring
│   ├── data/                      # VisDrone loader, ground truth
│   ├── prompts/                   # Prompt template management
│   └── utils/                     # Config loader, GPU detection
├── tests/                         # Foundation + integration tests
└── results/                       # Output (logs, cache, analysis charts)
```
