"""
Phase Timing Visualization - Gantt-style chart showing processing phases
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_phase_timing(log_dir: str = "results/logs") -> pd.DataFrame:
    """Load phase timing data from JSONL logs"""
    log_path = Path(log_dir)
    rows = []
    
    for log_file in log_path.glob("run_*.jsonl"):
        with open(log_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                entry = pd.json_normalize([json.loads(line)]).iloc[0]
                
                if entry.get('event_type') == 'phase_timing':
                    rows.append({
                        "timestamp": entry.get("timestamp"),
                        "run_id": entry.get("run_id"),
                        "image_id": entry.get("image_id"),
                        "pipeline": entry.get("pipeline"),
                        "repetition": entry.get("repetition", 0),
                        "vision_ms": entry.get("data.vision_ms", 0),
                        "llm_atomic_ms": entry.get("data.llm_atomic_ms", 0),
                        "llm_composite_ms": entry.get("data.llm_composite_ms", 0),
                        "total_ms": entry.get("data.total_ms", 0),
                        "individual_llm_times": entry.get("data.individual_llm_times", []),
                    })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_phase_timing_gantt(df: pd.DataFrame, output_dir: str = "results/analysis"):
    """
    Generate Gantt-style phase timing visualization.
    
    Structure:
    - Y-axis: Pipeline groups with runs as rows
    - X-axis: Time in ms (aligned phases)
    - Bars: Vision | LLM-Atomic | LLM-Composite
    
    Phase alignment:
    - All Vision bars start at x=0
    - All LLM-Atomic bars start at max vision time + buffer
    - All LLM-Composite bars start at max LLM-Atomic time + buffer
    """
    import json  # For json_normalize
    
    if df.empty:
        print("  ⚠️  No phase timing data available")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Color scheme for phases
    colors = {
        'vision': '#3498db',      # Blue
        'llm_atomic': '#e74c3c',  # Red
        'llm_composite': '#2ecc71'  # Green
    }
    
    # Calculate phase alignment offsets
    # Find max times across all runs for alignment
    max_vision = df['vision_ms'].max()
    max_atomic = df['llm_atomic_ms'].max()
    
    # Phase start positions (aligned at same x-offset per phase)
    phase_buffer = 20  # ms between phase groups
    phase_offsets = {
        'vision': 0,
        'llm_atomic': max_vision + phase_buffer,
        'llm_composite': max_vision + max_atomic + phase_buffer * 2
    }
    
    # Calculate total width needed
    max_total = max_vision + max_atomic + df['llm_composite_ms'].max() + phase_buffer * 3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max_total / 50, len(df) * 0.8 + 4))
    
    # Sort by pipeline and image
    df = df.sort_values(['pipeline', 'image_id', 'repetition'])
    
    # Create y-positions for each run
    y_positions = range(len(df))
    
    # Draw bars for each phase
    bar_height = 0.6
    
    for idx, (_, row) in enumerate(df.iterrows()):
        y = len(df) - idx - 1  # Flip so first row is at top
        
        # Vision bar
        ax.barh(y, row['vision_ms'], height=bar_height, 
                left=phase_offsets['vision'], 
                color=colors['vision'], 
                edgecolor='white', linewidth=0.5)
        
        # LLM Atomic bar
        ax.barh(y, row['llm_atomic_ms'], height=bar_height,
                left=phase_offsets['llm_atomic'],
                color=colors['llm_atomic'],
                edgecolor='white', linewidth=0.5)
        
        # LLM Composite bar
        ax.barh(y, row['llm_composite_ms'], height=bar_height,
                left=phase_offsets['llm_composite'],
                color=colors['llm_composite'],
                edgecolor='white', linewidth=0.5)
    
    # Create pipeline labels on y-axis
    y_labels = []
    for idx, (_, row) in enumerate(df.iterrows()):
        # Short label: pipeline + image_id + rep
        label = f"{row['pipeline']} | {row['image_id'][-13:]} | R{row['repetition']+1}"
        y_labels.append(label)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Add phase labels at top
    ax.axvline(x=phase_offsets['llm_atomic'] - phase_buffer/2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=phase_offsets['llm_composite'] - phase_buffer/2, color='gray', linestyle='--', alpha=0.5)
    
    # Phase header labels
    ax.text(phase_offsets['vision'] + max_vision/2, len(df) + 0.5, 'Vision Processing', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors['vision'])
    ax.text(phase_offsets['llm_atomic'] + max_atomic/2, len(df) + 0.5, 'LLM (Atomic)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors['llm_atomic'])
    ax.text(phase_offsets['llm_composite'] + df['llm_composite_ms'].max()/2, len(df) + 0.5, 'LLM (Composite)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors['llm_composite'])
    
    # X-axis formatting
    ax.set_xlabel('Time (ms)', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['vision'], label=f'Vision (avg: {df["vision_ms"].mean():.1f}ms)'),
        Patch(facecolor=colors['llm_atomic'], label=f'LLM Atomic (avg: {df["llm_atomic_ms"].mean():.1f}ms)'),
        Patch(facecolor=colors['llm_composite'], label=f'LLM Composite (avg: {df["llm_composite_ms"].mean():.1f}ms)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Title
    ax.set_title('Processing Phase Timing (Gantt Chart)', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust limits
    ax.set_xlim(-10, max_total + 50)
    ax.set_ylim(-1, len(df) + 1.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'phase_timing_gantt.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path / 'phase_timing_gantt.png'}")


def generate_phase_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for phase timing"""
    if df.empty:
        return {}
    
    return {
        "by_pipeline": {
            pipeline: {
                "vision_mean": data['vision_ms'].mean(),
                "vision_std": data['vision_ms'].std(),
                "llm_atomic_mean": data['llm_atomic_ms'].mean(),
                "llm_atomic_std": data['llm_atomic_ms'].std(),
                "llm_composite_mean": data['llm_composite_ms'].mean(),
                "llm_composite_std": data['llm_composite_ms'].std(),
                "total_mean": data['total_ms'].mean(),
                "count": len(data)
            }
            for pipeline, data in df.groupby('pipeline')
        },
        "overall": {
            "vision_mean": df['vision_ms'].mean(),
            "llm_atomic_mean": df['llm_atomic_ms'].mean(),
            "llm_composite_mean": df['llm_composite_ms'].mean(),
            "total_mean": df['total_ms'].mean(),
            "count": len(df)
        }
    }