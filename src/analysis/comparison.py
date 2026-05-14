"""
Analysis and comparison utilities
"""
import json
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd


def load_results(log_dir: str = "results/logs") -> pd.DataFrame:
    """
    Load all results from JSONL log files into a flat DataFrame.
    Handles multiple event types: vision_metrics, vision_resources, llm_resources, llm_tokens
    """
    log_path = Path(log_dir)
    rows = []
    
    for log_file in log_path.glob("run_*.jsonl"):
        with open(log_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                entry = json.loads(line)
                event_type = entry.get("event_type")
                data = entry.get("data", {})
                
                # Create base row
                row = {
                    "timestamp": entry.get("timestamp"),
                    "run_id": entry.get("run_id"),
                    "image_id": entry.get("image_id"),
                    "pipeline": entry.get("pipeline"),
                    "event_type": event_type,
                    "repetition": entry.get("repetition", 0),
                }
                
                # Extract vision metrics (includes predictions AND ground truth)
                if event_type == "vision_metrics":
                    row.update({
                        "cars": data.get("cars"),
                        "pedestrians": data.get("pedestrians"),
                        "bicycles": data.get("bicycles"),
                        "total_detections": data.get("total_detections"),
                        "avg_confidence": data.get("avg_confidence"),
                        "vision_latency_ms": data.get("latency_ms"),
                        "cache_hit": data.get("cache_hit"),
                    })
                    # Flatten avg_confidence_per_class
                    conf_per_class = data.get("avg_confidence_per_class", {})
                    for cls, conf in conf_per_class.items():
                        row[f"confidence_{cls}"] = conf
                    
                    # Extract ground truth if available (logged as separate keys)
                    row.update({
                        "gt_cars": data.get("gt_cars"),
                        "gt_pedestrians": data.get("gt_pedestrians"),
                        "gt_bicycles": data.get("gt_bicycles"),
                    })
                
                # Extract vision-only resources
                elif event_type == "vision_resources":
                    summary = data.get("summary", {})
                    row.update({
                        "vision_sample_count": summary.get("sample_count"),
                        "vision_duration_ms": summary.get("duration_ms"),
                        "vision_cpu_avg": summary.get("cpu_avg"),
                        "vision_cpu_max": summary.get("cpu_max"),
                        "vision_cpu_min": summary.get("cpu_min"),
                        "vision_memory_avg": summary.get("memory_avg"),
                        "vision_memory_max": summary.get("memory_max"),
                        "vision_memory_min": summary.get("memory_min"),
                        "vision_process_cpu_avg": summary.get("process_cpu_avg"),
                        "vision_process_cpu_max": summary.get("process_cpu_max"),
                    })
                
                # Extract LLM-only resources
                elif event_type == "llm_resources":
                    summary = data.get("summary", {})
                    row.update({
                        "llm_sample_count": summary.get("sample_count"),
                        "llm_duration_ms": summary.get("duration_ms"),
                        "llm_cpu_avg": summary.get("cpu_avg"),
                        "llm_cpu_max": summary.get("cpu_max"),
                        "llm_cpu_min": summary.get("cpu_min"),
                        "llm_memory_avg": summary.get("memory_avg"),
                        "llm_memory_max": summary.get("memory_max"),
                        "llm_memory_min": summary.get("memory_min"),
                        "llm_process_cpu_avg": summary.get("process_cpu_avg"),
                        "llm_process_cpu_max": summary.get("process_cpu_max"),
                    })
                
                # Extract LLM tokens
                elif event_type == "llm_tokens":
                    row.update({
                        "input_tokens": data.get("input_tokens"),
                        "output_tokens": data.get("output_tokens"),
                        "total_tokens": data.get("total_tokens"),
                    })
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis"""
    rows = []
    
    for r in results:
        row = {
            'image_id': r.get('image_id'),
            'pipeline': r.get('pipeline'),
            'repetition': r.get('repetition', 0),
            'vision_latency_ms': r.get('vision_latency_ms', 0),
            'llm_latency_ms': r.get('llm_latency_ms', 0),
            'total_latency_ms': r.get('total_latency_ms', 0),
            'vision_cache_hit': r.get('vision_cache_hit', False),
        }
        
        # Ground truth
        gt = r.get('ground_truth', {})
        row['gt_cars'] = gt.get('cars', 0)
        row['gt_pedestrians'] = gt.get('pedestrians', 0)
        row['gt_bicycles'] = gt.get('bicycles', 0)
        
        # Predictions from IR
        ir = r.get('intermediate_representation', {})
        row['pred_cars'] = ir.get('cars', 0)
        row['pred_pedestrians'] = ir.get('pedestrians', 0)
        row['pred_bicycles'] = ir.get('bicycles', 0)
        
        # Resource usage
        resources = r.get('resources', {})
        row['cpu_percent'] = resources.get('cpu_percent', 0)
        row['memory_mb'] = resources.get('memory_used_mb', 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_pipelines(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare performance across pipelines"""
    comparison = {}
    
    for pipeline in df['pipeline'].unique():
        pipe_data = df[df['pipeline'] == pipeline]
        
        comparison[pipeline] = {
            'avg_total_latency_ms': pipe_data['total_latency_ms'].mean(),
            'std_total_latency_ms': pipe_data['total_latency_ms'].std(),
            'avg_vision_latency_ms': pipe_data['vision_latency_ms'].mean(),
            'avg_llm_latency_ms': pipe_data['llm_latency_ms'].mean(),
            'avg_cpu_percent': pipe_data['cpu_percent'].mean(),
            'avg_memory_mb': pipe_data['memory_mb'].mean(),
            'num_samples': len(pipe_data),
            'cache_hit_rate': pipe_data['vision_cache_hit'].mean() if 'vision_cache_hit' in pipe_data else 0
        }
    
    return comparison


def save_comparison(comparison: Dict[str, Any], output_path: str = "results/pipeline_comparison.json"):
    """Save comparison results to JSON"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)


def generate_visualizations(df: pd.DataFrame, output_dir: str = "results/analysis"):
    """
    Generate publication-ready plots for research paper.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # 1. Vision Latency Comparison Boxplot
    vision_df = df[df['event_type'] == 'vision_metrics'].copy()
    if not vision_df.empty and 'vision_latency_ms' in vision_df.columns:
        vision_df['vision_latency_ms'] = pd.to_numeric(vision_df['vision_latency_ms'], errors='coerce')
        vision_df = vision_df.dropna(subset=['vision_latency_ms'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=vision_df, x='pipeline', y='vision_latency_ms', ax=ax)
        ax.set_title('Vision Processing Latency by Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Pipeline', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'vision_latency_comparison.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'vision_latency_comparison.png'}")
    
    # 2. Detection Count Comparison with Ground Truth
    if not vision_df.empty:
        detection_cols = ['cars', 'pedestrians', 'bicycles']
        for col in detection_cols:
            vision_df[col] = pd.to_numeric(vision_df[col], errors='coerce')
        
        # Get mean predictions and ground truth per pipeline
        # FIX: Already using mean() - this was correct
        detection_summary = vision_df.groupby('pipeline')[detection_cols].mean()
        
        # Also calculate ground truth means
        gt_cols = ['gt_cars', 'gt_pedestrians', 'gt_bicycles']
        for col in gt_cols:
            if col in vision_df.columns:
                vision_df[col] = pd.to_numeric(vision_df[col], errors='coerce')
        
        gt_summary = vision_df.groupby('pipeline')[['gt_cars', 'gt_pedestrians', 'gt_bicycles']].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Predictions
        detection_summary.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#3498db', '#2ecc71'])
        axes[0].set_title('Predicted Counts', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xlabel('Pipeline', fontsize=12)
        axes[0].legend(title='Object Type', fontsize=10)
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        # Right: Ground Truth
        gt_summary.columns = ['cars', 'pedestrians', 'bicycles']  # Rename for plotting
        gt_summary.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.6)
        axes[1].set_title('Ground Truth Counts', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_xlabel('Pipeline', fontsize=12)
        axes[1].legend(title='Object Type', fontsize=10)
        plt.sca(axes[1])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'detection_counts.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'detection_counts.png'}")
    
    # 3. Confidence Score Comparison
    if not vision_df.empty and 'avg_confidence' in vision_df.columns:
        vision_df['avg_confidence'] = pd.to_numeric(vision_df['avg_confidence'], errors='coerce')
        conf_df = vision_df.dropna(subset=['avg_confidence'])
        
        if not conf_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=conf_df, x='pipeline', y='avg_confidence', ax=ax)
            ax.set_title('Average Confidence Score by Pipeline', fontsize=14, fontweight='bold')
            ax.set_ylabel('Confidence', fontsize=12)
            ax.set_xlabel('Pipeline', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'confidence_comparison.png', dpi=300)
            plt.close()
            print(f"  Saved: {output_path / 'confidence_comparison.png'}")
    
    # 4. Vision Resource Usage (Bar Charts + Time Series)
    vision_res_df = df[df['event_type'] == 'vision_resources'].copy()
    if not vision_res_df.empty:
        # 4a. Bar Charts for Average CPU and Memory
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Vision Processing Resource Usage (Average per Pipeline)', fontsize=14, fontweight='bold')
        
        # Get unique pipelines
        pipelines = vision_res_df['pipeline'].unique()
        x = range(len(pipelines))
        
        # Average System CPU Usage
        cpu_avgs = []
        cpu_stds = []
        for pipe in pipelines:
            pipe_data = vision_res_df[vision_res_df['pipeline'] == pipe]
            cpu_vals = pd.to_numeric(pipe_data['vision_cpu_avg'], errors='coerce').dropna()
            if len(cpu_vals) > 0:
                cpu_avgs.append(cpu_vals.mean())
                cpu_stds.append(cpu_vals.std() if len(cpu_vals) > 1 else 0)
            else:
                cpu_avgs.append(0)
                cpu_stds.append(0)
        
        axes[0].bar(x, cpu_avgs, yerr=cpu_stds, capsize=5, color='#3498db', alpha=0.8)
        axes[0].set_title('System: Average CPU Usage', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('CPU %', fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pipelines, rotation=45, ha='right')
        axes[0].set_xlabel('What it means: Total CPU usage across ALL cores\naveraged over the monitoring period')
        
        # Average Memory Usage
        mem_avgs = []
        mem_stds = []
        for pipe in pipelines:
            pipe_data = vision_res_df[vision_res_df['pipeline'] == pipe]
            mem_vals = pd.to_numeric(pipe_data['vision_memory_avg'], errors='coerce').dropna()
            if len(mem_vals) > 0:
                mem_avgs.append(mem_vals.mean())
                mem_stds.append(mem_vals.std() if len(mem_vals) > 1 else 0)
            else:
                mem_avgs.append(0)
                mem_stds.append(0)
        
        axes[1].bar(x, mem_avgs, yerr=mem_stds, capsize=5, color='#2ecc71', alpha=0.8)
        axes[1].set_title('System: Average Memory Usage', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Memory %', fontsize=10)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pipelines, rotation=45, ha='right')
        axes[1].set_xlabel('What it means: Percentage of total system RAM\nused during vision processing')
        
        # Process CPU Usage
        proc_avgs = []
        proc_stds = []
        for pipe in pipelines:
            pipe_data = vision_res_df[vision_res_df['pipeline'] == pipe]
            proc_vals = pd.to_numeric(pipe_data['vision_process_cpu_avg'], errors='coerce').dropna()
            if len(proc_vals) > 0:
                proc_avgs.append(proc_vals.mean())
                proc_stds.append(proc_vals.std() if len(proc_vals) > 1 else 0)
            else:
                proc_avgs.append(0)
                proc_stds.append(0)
        
        axes[2].bar(x, proc_avgs, yerr=proc_stds, capsize=5, color='#e74c3c', alpha=0.8)
        axes[2].set_title('Process: Average CPU Usage', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('CPU %', fontsize=10)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(pipelines, rotation=45, ha='right')
        axes[2].set_xlabel('What it means: CPU usage of ONLY the vision\nprocess itself (can exceed 100% on multi-core)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'vision_resource_bars.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'vision_resource_bars.png'}")
        
        # 4b. Time Series (existing)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Vision Processing Resource Usage (Time Series)', fontsize=16, fontweight='bold')
        
        # Vision CPU usage
        cpu_data = vision_res_df.dropna(subset=['vision_cpu_avg'])
        if not cpu_data.empty:
            for pipeline in cpu_data['pipeline'].unique():
                pipe_data = cpu_data[cpu_data['pipeline'] == pipeline]
                axes[0,0].plot(pipe_data.index, pipe_data['vision_cpu_avg'], label=pipeline)
            axes[0,0].set_title('Vision: Average CPU Usage', fontsize=12)
            axes[0,0].set_ylabel('CPU %', fontsize=10)
            axes[0,0].legend(fontsize=9)
        
        # Vision Memory usage
        mem_data = vision_res_df.dropna(subset=['vision_memory_avg'])
        if not mem_data.empty:
            for pipeline in mem_data['pipeline'].unique():
                pipe_data = mem_data[mem_data['pipeline'] == pipeline]
                axes[0,1].plot(pipe_data.index, pipe_data['vision_memory_avg'], label=pipeline)
            axes[0,1].set_title('Vision: Average Memory Usage', fontsize=12)
            axes[0,1].set_ylabel('Memory %', fontsize=10)
            axes[0,1].legend(fontsize=9)
        
        # Vision Process CPU
        proc_data = vision_res_df.dropna(subset=['vision_process_cpu_avg'])
        if not proc_data.empty:
            for pipeline in proc_data['pipeline'].unique():
                pipe_data = proc_data[proc_data['pipeline'] == pipeline]
                axes[1,0].plot(pipe_data.index, pipe_data['vision_process_cpu_avg'], label=pipeline)
            axes[1,0].set_title('Vision: Process CPU Usage', fontsize=12)
            axes[1,0].set_ylabel('Process CPU %', fontsize=10)
            axes[1,0].legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'vision_resources.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'vision_resources.png'}")
    
    # 5. LLM Resource Usage (separate from Vision)
    llm_res_df = df[df['event_type'] == 'llm_resources'].copy()
    if not llm_res_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LLM Processing Resource Usage', fontsize=16, fontweight='bold')
        
        # LLM CPU usage
        cpu_data = llm_res_df.dropna(subset=['llm_cpu_avg'])
        if not cpu_data.empty:
            for pipeline in cpu_data['pipeline'].unique():
                pipe_data = cpu_data[cpu_data['pipeline'] == pipeline]
                axes[0,0].plot(pipe_data.index, pipe_data['llm_cpu_avg'], label=pipeline)
            axes[0,0].set_title('LLM: Average CPU Usage', fontsize=12)
            axes[0,0].set_ylabel('CPU %', fontsize=10)
            axes[0,0].legend(fontsize=9)
        
        # LLM Memory usage
        mem_data = llm_res_df.dropna(subset=['llm_memory_avg'])
        if not mem_data.empty:
            for pipeline in mem_data['pipeline'].unique():
                pipe_data = mem_data[mem_data['pipeline'] == pipeline]
                axes[0,1].plot(pipe_data.index, pipe_data['llm_memory_avg'], label=pipeline)
            axes[0,1].set_title('LLM: Average Memory Usage', fontsize=12)
            axes[0,1].set_ylabel('Memory %', fontsize=10)
            axes[0,1].legend(fontsize=9)
        
        # LLM Process CPU
        proc_data = llm_res_df.dropna(subset=['llm_process_cpu_avg'])
        if not proc_data.empty:
            for pipeline in proc_data['pipeline'].unique():
                pipe_data = proc_data[proc_data['pipeline'] == pipeline]
                axes[1,0].plot(pipe_data.index, pipe_data['llm_process_cpu_avg'], label=pipeline)
            axes[1,0].set_title('LLM: Process CPU Usage', fontsize=12)
            axes[1,0].set_ylabel('Process CPU %', fontsize=10)
            axes[1,0].legend(fontsize=9)
        
        # Token usage
        token_df = df[df['event_type'] == 'llm_tokens'].copy()
        if not token_df.empty and 'total_tokens' in token_df.columns:
            token_df['total_tokens'] = pd.to_numeric(token_df['total_tokens'], errors='coerce')
            token_summary = token_df.groupby('pipeline')['total_tokens'].mean()
            
            axes[1,1].bar(token_summary.index, token_summary.values)
            axes[1,1].set_title('Average Token Usage', fontsize=12)
            axes[1,1].set_ylabel('Tokens', fontsize=10)
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'llm_resources.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'llm_resources.png'}")
    
    print(f"\n✅ All visualizations saved to {output_path}")
    
    # Add accuracy charts (Option 1)
    generate_accuracy_charts(df, output_dir)


def generate_accuracy_charts(df: pd.DataFrame, output_dir: str = "results/analysis"):
    """
    Generate accuracy visualization charts (Option 1).
    Chart 1: Predicted vs Actual (Grouped Bar Chart) per category
    Chart 2: MAE Heatmap - pipeline vs category
    """
    from src.evaluation.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data - filter to vision_metrics events that have ground truth
    vision_df = df[df['event_type'] == 'vision_metrics'].copy()
    
    # Check if ground truth columns exist
    gt_cols = ['gt_cars', 'gt_pedestrians', 'gt_bicycles']
    pred_cols = ['cars', 'pedestrians', 'bicycles']
    
    # Convert to numeric
    for col in gt_cols + pred_cols:
        if col in vision_df.columns:
            vision_df[col] = pd.to_numeric(vision_df[col], errors='coerce')
    
    # Filter rows that have ground truth
    gt_available = vision_df.dropna(subset=gt_cols, how='any')
    
    if gt_available.empty:
        print("  ⚠️  No ground truth data available for accuracy charts")
        return
    
    # Chart 1: Predicted vs Actual (Grouped Bar Chart with Ground Truth Section)
    # FIX: Use mean instead of sum to avoid counting same image multiple times
    categories = ['cars', 'pedestrians', 'bicycles']
    display_names = ['Cars', 'Pedestrians', 'Bicycles']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('Predicted vs Actual Counts by Pipeline', fontsize=16, fontweight='bold')
    
    # First axis: Ground Truth bar for each pipeline (mean per unique image)
    ax_gt = axes[0]
    gt_plot_data = []
    for pipeline in gt_available['pipeline'].unique():
        pipe_data = gt_available[gt_available['pipeline'] == pipeline]
        for cat, display in zip(categories, display_names):
            gt_col = f'gt_{cat}'
            if gt_col in pipe_data.columns:
                # Use mean to avoid duplicate counting from repetitions
                gt_plot_data.append({
                    'pipeline': pipeline,
                    'category': display,
                    'gt_count': pipe_data[gt_col].mean()
                })
    
    if gt_plot_data:
        gt_df = pd.DataFrame(gt_plot_data)
        gt_df['type'] = 'Ground Truth'
        sns.barplot(data=gt_df, x='category', y='gt_count', hue='pipeline', ax=ax_gt, palette='Greys')
        ax_gt.set_title('Ground Truth\n(Reference)', fontsize=12, fontweight='bold')
        ax_gt.set_xlabel('Category', fontsize=10)
        ax_gt.set_ylabel('Count', fontsize=10)
        ax_gt.tick_params(axis='x', rotation=45)
        ax_gt.legend(fontsize=8, title='Pipeline')
    
    # Remaining axes: Actual vs Predicted per category
    for idx, (cat, display) in enumerate(zip(categories, display_names)):
        ax = axes[idx + 1]
        
        # Calculate mean predicted vs actual per pipeline (FIX: use mean, not sum)
        plot_data = []
        for pipeline in gt_available['pipeline'].unique():
            pipe_data = gt_available[gt_available['pipeline'] == pipeline]
            gt_col = f'gt_{cat}'
            if gt_col in pipe_data.columns:
                plot_data.append({
                    'pipeline': pipeline,
                    'type': 'Actual',
                    'count': pipe_data[gt_col].mean(),  # FIX: use mean
                    'category': display
                })
            plot_data.append({
                'pipeline': pipeline,
                'type': 'Predicted',
                'count': pipe_data[cat].mean(),  # FIX: use mean
                'category': display
            })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.barplot(data=plot_df, x='pipeline', y='count', hue='type', ax=ax, palette={'Actual': '#3498db', 'Predicted': '#e74c3c'})
            ax.set_title(f'{display}', fontsize=12)
            ax.set_xlabel('Pipeline', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_predicted_vs_actual.png', dpi=300)
    plt.close()
    print(f"  Saved: {output_path / 'accuracy_predicted_vs_actual.png'}")
    
    # Chart 2: MAE Heatmap
    heatmap_data = []
    
    for pipeline in gt_available['pipeline'].unique():
        pipe_data = gt_available[gt_available['pipeline'] == pipeline]
        
        for cat in categories:
            y_true = pipe_data[f'gt_{cat}'].dropna().tolist()
            y_pred = pipe_data[cat].dropna().tolist()
            
            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            if min_len > 0:
                mae = mean_absolute_error(y_true[:min_len], y_pred[:min_len])
                heatmap_data.append({
                    'pipeline': pipeline,
                    'category': cat,
                    'mae': mae
                })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='pipeline', columns='category', values='mae')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Mean Absolute Error (MAE) by Pipeline and Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Pipeline', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_mae_heatmap.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'accuracy_mae_heatmap.png'}")
    
    # Chart 3: Accuracy Comparison (Ground Truth vs Predictions stacked bar)
    # Show ground truth alongside predictions for comparison
    # FIX: Use mean instead of sum to avoid duplicate counting from repetitions
    error_data = []
    
    for pipeline in gt_available['pipeline'].unique():
        pipe_data = gt_available[gt_available['pipeline'] == pipeline]
        
        for cat, display in zip(categories, display_names):
            gt_col = f'gt_{cat}'
            pred_col = cat
            
            # FIX: Use mean() instead of sum() to get per-image average
            gt_total = pipe_data[gt_col].mean() if gt_col in pipe_data.columns else 0
            pred_total = pipe_data[pred_col].mean() if pred_col in pipe_data.columns else 0
            
            error_data.append({
                'pipeline': pipeline,
                'category': display,
                'Ground Truth': gt_total,
                'Predicted': pred_total
            })
    
    if error_data:
        error_df = pd.DataFrame(error_data)
        
        # Create grouped bar chart: Ground Truth vs Predicted
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Ground Truth vs Predicted Counts by Pipeline', fontsize=16, fontweight='bold')
        
        for idx, cat in enumerate(display_names):
            ax = axes[idx]
            cat_data = error_df[error_df['category'] == cat]
            
            x = cat_data['pipeline']
            gt_vals = cat_data['Ground Truth'].values
            pred_vals = cat_data['Predicted'].values
            
            x_pos = range(len(x))
            ax.bar([p - 0.2 for p in x_pos], gt_vals, width=0.4, label='Ground Truth', color='#34495e', alpha=0.8)
            ax.bar([p + 0.2 for p in x_pos], pred_vals, width=0.4, label='Predicted', color='#e74c3c', alpha=0.8)
            
            ax.set_title(f'{cat}', fontsize=14)
            ax.set_xlabel('Pipeline', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x, rotation=45)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_errors.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_path / 'accuracy_errors.png'}")
    
    print(f"✅ Accuracy charts saved to {output_path}")
