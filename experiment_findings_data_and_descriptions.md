# Experiment Findings: Drone Vision Model Benchmarks

**Run**: `run_1778689714.jsonl`
**Dataset**: 100 VisDrone images | **4 Pipelines** | **3 Repetitions** (cached after first run)
**Note**: All time-based charts use **fresh (non-cached) runs only** (1st repetition). Cached runs (2nd+3rd) complete in ~3ms and are excluded. Classification uses a latency threshold (>100ms for YOLO, >1000ms for VLM) since `cache_hit` field in the experiment code always logs as `False` (a bug).

---

## Architecture Overview

```
Image + Prompt
    ↓
Vision Processing (YOLO detection OR VLM image→text)
    ↓
Intermediate Representation (standardized JSON: cars, pedestrians, bicycles)
    ↓
Controller LLM (Gemma4:e4b via vLLM) — 3 atomic + 1 composite query
    ↓
Structured Output (counts + description)
```

**4 Pipelines:** `yolo_local`, `yolo_mcp`, `vlm_direct`, `vlm_multiagent`
**3 Metrics:** MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), Exact Match %

---

## Chart 1: Accuracy Metrics Comparison (MAE / RMSE / ExactMatch%)

**Description**: Three grouped bar charts side-by-side, one per metric (MAE, RMSE, ExactMatch%). Each chart shows all 4 pipelines on the x-axis with bars colored by category (red=cars, blue=pedestrians, green=bicycles). Value labels printed on bars.

| Pipeline | Cars MAE | Cars RMSE | Cars EM% | Peds MAE | Peds RMSE | Peds EM% | Bikes MAE | Bikes RMSE | Bikes EM% |
|---|---|---|---|---|---|---|---|---|---|
| **yolo_mcp** | **6.79** | 13.58 | **32.0%** | 23.48 | 43.06 | **25.0%** | 7.62 | 15.30 | 28.0% |
| **yolo_local** | 6.84 | 13.57 | 31.0% | 23.47 | 43.05 | 24.0% | 7.60 | 15.29 | 28.0% |
| vlm_direct | 9.27 | 14.93 | 20.0% | **19.24** | **38.02** | 23.0% | **6.87** | **13.68** | **29.0%** |
| vlm_multiagent | 12.61 | 21.52 | 29.0% | 22.31 | 39.95 | 20.0% | 9.23 | 18.57 | 27.0% |

**Key Finding**: YOLO variants are tied-best on cars (MAE ~6.8) and exact match (~31-32%). VLM direct is best on pedestrians (MAE 19.24) and bicycles (MAE 6.87). All pipelines **systematically underestimate** every category.

---

## Chart 2: MAE Heatmap (Pipeline × Category)

**Description**: Color-coded heatmap (YlOrRd colormap) with pipelines on the Y-axis and categories (cars, pedestrians, bicycles) on the X-axis. Darker red = worse MAE. Numeric values annotated in each cell.

| Pipeline | Cars | Pedestrians | Bicycles |
|---|---|---|---|
| vlm_direct | 9.27 | 19.24 | 6.87 |
| vlm_multiagent | 12.61 | 22.31 | 9.23 |
| yolo_local | 6.84 | 23.47 | 7.60 |
| yolo_mcp | 6.79 | 23.48 | 7.62 |

**Worst cell**: YOLO on pedestrians — MAE **23.47-23.48** (severe underestimation of dense crowds)
**Best cell**: vlm_direct on bicycles — MAE **6.87**
**Pattern**: YOLO excels at cars; VLM is better at pedestrians and bicycles

---

## Chart 3: Predicted vs Ground Truth Counts

**Description**: Three bar charts — one per category (Cars, Pedestrians, Bicycles). Each chart shows all 4 pipelines with grouped bars for Ground Truth (dark gray) and Predicted (red). Y-axis is average count per image.

| Category | GT Avg | yolo_local | yolo_mcp | vlm_direct | vlm_multiagent |
|---|---|---|---|---|---|
| Cars | 14.5 | 9.2 | 9.1 | 8.1 | 4.1 |
| Pedestrians | 29.6 | 6.2 | 6.1 | 13.1 | 13.7 |
| Bicycles | 8.3 | 1.2 | 1.2 | 2.1 | 2.3 |

**Finding**: Every pipeline dramatically undershoots. The gap is most severe for pedestrians — all pipelines predict <14 when GT averages ~30. YOLO is almost blind to pedestrians (predicts only 6.2 vs GT 29.6).

---

## Chart 4: Vision Latency Comparison (Boxplot)

**Description**: Single shared-axis boxplot with 4 pipelines ordered left-to-right (yolo_local, yolo_mcp, vlm_direct, vlm_multiagent). Y-axis is latency in milliseconds. No outlier dots (showfliers=False). Stats labels (μ/med/σ) positioned below each x-axis tick.

| Pipeline | Mean | Median | Std | Interpretation |
|---|---|---|---|---|
| **yolo_mcp** | **171ms** | **169ms** | ±47ms | Fastest vision. YOLOv8m via MCP server (no model load overhead). |
| yolo_local | 224ms | 205ms | ±150ms | Slightly slower due to in-process model loading for first image. High σ from first-image model init (~1,648ms) pulling mean up. |
| vlm_direct | 12,106ms | 11,967ms | ±1,621ms | 71× slower than yolo_mcp. Vision IS an LLM call — sends full image to Gemma4:e4b for scene analysis. Consistent variance (±1.6s). |
| vlm_multiagent | 32,758ms | 29,575ms | ±17,660ms | 191× slower than yolo_mcp. Multi-agent refinement iterates (initial analysis + refinement), doubling vision time. Very high σ from variable iteration count (up to 3 passes). |

**Finding**: YOLO vision is ~200ms; VLM vision takes 12-33 seconds — a **71-191×** difference. The yolo_local outlier at ~1,648ms (first image only) is model initialization. vlm_multiagent's massive σ (±17.7s) reflects variable refinement iterations.

---

## Chart 5: Phase Timing Aligned Gantt (Average per Pipeline)

**Description**: Phase-aligned horizontal bar chart with 4 rows (one per pipeline). Blue bars (Vision) all start at x=0. Red bars (LLM Atomic ×3) all start at `max_vision + 5s` buffer. Green bars (LLM Composite) all start after the atomic column. Column dividers (dashed vertical lines) separate the three phase columns. Black text labels show mean duration per bar.

| Pipeline | Vision (ms) | LLM Atomic (ms) | LLM Composite (ms) | **Total (ms)** |
|---|---|---|---|---|
| yolo_mcp | **171** ± 47 | 9,938 ± 3,000 | 7,072 ± 2,196 | **17,181** |
| yolo_local | 224 ± 150 | 11,003 ± 3,171 | 6,942 ± 1,997 | 18,169 |
| vlm_direct | 12,106 ± 1,621 | 15,314 ± 1,635 | 8,565 ± 1,391 | 35,985 |
| vlm_multiagent | 32,758 ± 17,660 | 17,383 ± 6,951 | 8,806 ± 3,781 | 58,947 |

**Finding**:
- **Vision phase**: The gulf between YOLO (~200ms) and VLM (12-33s) is the defining characteristic of the two approaches. YOLO does fast object detection; VLM does image-to-text via LLM.
- **LLM Atomic phase**: Atomic queries (3 per image: cars/pedestrians/bicycles) take ~10-11s for YOLO pipelines but ~15-17s for VLM. This is likely because VLM's IR text output is richer/longer, increasing prompt processing + response time.
- **LLM Composite phase**: The single composite query is the most consistent across pipelines (~7-9s), since it's the same query structure regardless of pipeline.
- **LLM is the bottleneck** in every pipeline: even for yolo_mcp (fastest total at 17.2s), 99% of execution time is the controller LLM.

---

## Chart 6: Per-Phase Bar Charts (Mean ± Std)

**Description**: Three separate horizontal bar charts arranged side-by-side, one per processing phase (Vision, LLM Atomic, LLM Composite). Each chart shows all 4 pipelines with colored bars (blue/red/green) and error bars representing ±1 standard deviation. Numeric labels (mean ± std) positioned to the right of each bar.

**Vision Phase** (blue):
| Pipeline | Mean ± Std |
|---|---|
| yolo_mcp | 171 ± 47ms |
| yolo_local | 224 ± 150ms |
| vlm_direct | 12,106 ± 1,621ms |
| vlm_multiagent | 32,758 ± 17,660ms |

**LLM Atomic Phase** (red):
| Pipeline | Mean ± Std |
|---|---|
| yolo_mcp | 9,938 ± 3,000ms |
| yolo_local | 11,003 ± 3,171ms |
| vlm_direct | 15,314 ± 1,635ms |
| vlm_multiagent | 17,383 ± 6,951ms |

**LLM Composite Phase** (green):
| Pipeline | Mean ± Std |
|---|---|
| yolo_local | 6,942 ± 1,997ms |
| yolo_mcp | 7,072 ± 2,196ms |
| vlm_direct | 8,565 ± 1,391ms |
| vlm_multiagent | 8,806 ± 3,781ms |

**Finding**: The vision phase separates YOLO from VLM by 1-2 orders of magnitude. The atomic and composite phases are dominated by LLM inference time regardless of pipeline — the slowest pipeline (vlm_multiagent) is only 1.7× slower than the fastest (yolo_mcp) for atomic, and 1.3× for composite.

---

## Chart 7: Per-Image Scatter (GT vs Predicted)

**Description**: Four subplots in a 2×2 grid, one per pipeline. Each subplot is a scatter plot with ground truth count on the x-axis and predicted count on the y-axis. Three colors per plot (red=cars, blue=pedestrians, green=bicycles). A dashed diagonal line (y=x) represents perfect prediction — points above the line are overcounts, points below are undercounts.

### Bias (avg undercount per category):

| Pipeline | Cars Bias | Peds Bias | Bikes Bias |
|---|---|---|---|
| yolo_local | 5.4 under | 23.4 under | 7.2 under |
| yolo_mcp | 5.4 under | 23.4 under | 7.2 under |
| vlm_direct | 6.4 under | 16.5 under | 6.2 under |
| vlm_multiagent | 10.4 under | 15.9 under | 6.0 under |

**Finding**: Points cluster **below** the diagonal line across all pipelines, confirming systematic undercounting. Pedestrians show the widest spread and worst accuracy. Cars cluster somewhat near the diagonal for YOLO. VLM pipelines have a hard upper limit — they rarely predict >50 cars even when GT reaches 77.

---

## Chart 8: Confidence Score Comparison (YOLO only)

**Description**: Single boxplot comparing average detection confidence scores between yolo_local and yolo_mcp. VLM pipelines are excluded because they output text, not detection boxes, so they don't produce confidence scores.

| Pipeline | Avg Confidence | Std |
|---|---|---|
| yolo_local | 0.514 | ±0.106 |
| yolo_mcp | ~0.514 | ~±0.106 |

**Finding**: Average detection confidence is ~51%. The YOLOv8m model is COCO-trained (not VisDrone-finetuned), which explains the many low-confidence detections and poor recall on drone-perspective objects.

---

## Overall Summary

| Metric | Best Pipeline | Value |
|---|---|---|
| Cars MAE | yolo_mcp | **6.79** |
| Cars Exact Match | yolo_mcp | **32.0%** |
| Pedestrians MAE | vlm_direct | **19.24** |
| Pedestrians Exact Match | yolo_mcp | **25.0%** |
| Bicycles MAE | vlm_direct | **6.87** |
| Bicycles Exact Match | vlm_direct | **29.0%** |
| Fastest Vision | yolo_mcp | **171ms** |
| Fastest Total | yolo_mcp | **17.2s** |
| Lowest Bias (Cars) | yolo_mcp / yolo_local | **5.4 under** |
| Lowest Bias (Peds) | vlm_multiagent | **15.9 under** |
| Lowest Bias (Bikes) | vlm_multiagent | **6.0 under** |

### Key Takeaways

1. **No pipeline achieves acceptable accuracy** — the best exact match rate is only 32% (yolo_mcp on cars). Pedestrian MAE ranges 19-23, which is unacceptably high.
2. **Systematic undercounting** — every pipeline underestimates every category. This suggests the YOLO model (COCO-trained, not VisDrone-finetuned) misses many objects, and VLM models have a low ceiling on what they'll output.
3. **LLM is the bottleneck** — controller LLM takes ~17-18s per image even with YOLO pipelines, versus ~200ms for vision. Total runtime is 17-59s per image.
4. **Caching works** — second and third repetitions complete in ~3ms via cache.
5. **Data contradiction** - VLM Direct has lower MAE (19.24) than YOLO (23.47) on pedestrians despite YOLO being purpose-built for object detection. This may indicate the COCO-trained YOLO is poorly suited for drone-perspective pedestrian detection.

---

## Charts Generated

All charts saved to `results/analysis/`:

| File | Description |
|---|---|
| `accuracy_metrics_comparison.png` | MAE, RMSE, ExactMatch% grouped bar chart |
| `accuracy_mae_heatmap.png` | MAE heatmap (pipeline × category) |
| `accuracy_predicted_vs_actual.png` | Ground truth vs predicted per category |
| `accuracy_scatter.png` | Per-image scatter plots |
| `vision_latency_comparison.png` | Vision latency boxplot (fresh only, no outliers) |
| `phase_timing_gantt_aligned.png` | Phase-aligned average Gantt chart |
| `phase_timing_per_phase.png` | Per-phase bar charts with std |
| `confidence_comparison.png` | Confidence score boxplot (YOLO only) |
