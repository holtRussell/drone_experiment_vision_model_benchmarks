# Experiment Phases and Descriptions

## Phase 0: Config Loading

Loads `configs/experiment.yaml`, `model.yaml`, `prompts.yaml` — sets dataset path, model (Gemma4:e4b), number of images, which pipelines to run, repetitions, and prompt templates.

## Phase 1: Image Loading

`VisDroneLoader` iterates over `data/visdrone/images/*.jpg`, yielding dicts with `image_id`, `image_path`, and `annotation_path`. Images are resized to 640px max dimension.

## Phase 2: Ground Truth Parsing

For each image, `VisDroneGroundTruth.parse_annotation()` reads the VisDrone `.txt` label file and maps its 10 object categories down to 3: `{"cars": N, "pedestrians": N, "bicycles": N}`. This is the reference standard for all accuracy comparisons.

## Phase 3: Vision Processing → Intermediate Representation

Each pipeline runs on the image and produces an IR dict with the same 3 count fields. This is where the different vision approaches diverge:

- **YOLO pipelines**: Traditional object detection → structured counts
- **VLM pipelines**: LLM sees raw image → text response → parser extracts counts into IR

Both produce the same `{cars, pedestrians, bicycles}` schema with metadata (confidence scores, detections, latency).

## Phase 4: Controller LLM (IR → Structured Output)

The IR is fed into Gemma4:e4b via vLLM. Three **atomic queries** are asked (one per object type: "How many cars are there?") plus one **composite query**. The LLM reasons about the IR numbers and produces final structured responses.

## Phase 5: Evaluation / Accuracy Measurement

Calculates 3 metrics per category (+ overall) by comparing predicted counts against ground truth:

| Metric | What it measures |
|---|---|
| **MAE** | Average absolute error per image (e.g., "off by 2.3 cars on average") |
| **RMSE** | Root-mean-squared error — penalizes large outliers more heavily |
| **Exact Match %** | What fraction of predictions hit the ground truth exactly |

These are computed across all images for each of `cars`, `pedestrians`, `bicycles`, and `overall` (all categories concatenated).

## Phase 6: Analysis & Visualization

Post-hoc scripts in `src/analysis/comparison.py` produce:
1. Accuracy bar charts (predicted vs actual per category per pipeline)
2. MAE heatmap (pipeline × category)
3. Latency boxplots
4. Phase-timing Gantt charts
5. Resource usage plots (CPU/memory)

All results are stored as structured JSONL logs in `results/logs/`.
