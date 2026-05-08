#!/usr/bin/env python
"""
Analysis script for experiment results
Generates visualizations and summary statistics for research paper

Usage:
    python scripts/analyze_results.py [--log-dir results/logs] [--output-dir results/analysis]
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import load_results, generate_visualizations


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--log-dir', default='results/logs', 
                        help='Directory containing JSONL log files')
    parser.add_argument('--output-dir', default='results/analysis',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Experiment Results Analysis")
    print("=" * 50)
    
    # Load results
    print(f"\nLoading results from {args.log_dir}/...")
    df = load_results(args.log_dir)
    print(f"Loaded {len(df)} records")
    
    if df.empty:
        print("❌ No data found. Run an experiment first!")
        return
    
    # Show event types
    print("\nEvent types:")
    for etype, count in df['event_type'].value_counts().items():
        print(f"  {etype}: {count}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations in {args.output_dir}/...")
    generate_visualizations(df, args.output_dir)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics")
    print("=" * 50)
    
    # Vision metrics summary
    vision_df = df[df['event_type'] == 'vision_metrics'].copy()
    if not vision_df.empty and 'vision_latency_ms' in vision_df.columns:
        vision_df['vision_latency_ms'] = vision_df['vision_latency_ms'].apply(
            lambda x: float(x) if x is not None else None
        )
        vision_df_clean = vision_df.dropna(subset=['vision_latency_ms'])
        
        if not vision_df_clean.empty:
            print("\nVision Latency (ms):")
            print(vision_df_clean.groupby('pipeline')['vision_latency_ms'].agg(['mean', 'std', 'count']))
    
    # Token usage summary
    token_df = df[df['event_type'] == 'llm_tokens'].copy()
    if not token_df.empty and 'total_tokens' in token_df.columns:
        token_df['total_tokens'] = token_df['total_tokens'].apply(
            lambda x: float(x) if x is not None else None
        )
        token_clean = token_df.dropna(subset=['total_tokens'])
        
        if not token_clean.empty:
            print("\nLLM Token Usage:")
            print(token_clean.groupby('pipeline')['total_tokens'].agg(['mean', 'std', 'count']))
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
