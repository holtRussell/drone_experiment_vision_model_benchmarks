"""
Analysis and comparison utilities
"""
import json
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd


def load_results(log_dir: str = "results/logs") -> List[Dict[str, Any]]:
    """Load all results from JSONL log files"""
    log_path = Path(log_dir)
    results = []
    
    for log_file in log_path.glob("run_*.jsonl"):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    
    return results


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
