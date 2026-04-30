"""
Main entry point for the benchmark
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_all_configs
from src.experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run vision pipeline benchmark')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml',
                        help='Path to experiment config file')
    parser.add_argument('--model-config', type=str, default='configs/model.yaml',
                        help='Path to model config file')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Override number of images to process')
    parser.add_argument('--pipelines', type=str, nargs='+', default=None,
                        help='Pipelines to run (yolo_local, yolo_mcp, vlm_direct, vlm_multiagent)')
    
    args = parser.parse_args()
    
    # Load configs
    print("Loading configurations...")
    configs = load_all_configs()
    
    # Flatten the structure (experiment.yaml has nested 'experiment:' key)
    def flatten_config(config, key):
        if key in config and isinstance(config[key], dict) and key in config[key]:
            return config[key][key]
        return config.get(key, {})
    
    # Override with command line args
    if args.num_images:
        configs['experiment']['num_images'] = args.num_images
    
    if args.pipelines:
        configs['experiment']['pipelines'] = args.pipelines
    
    # Build full_config for the runner
    full_config = {
        'model': flatten_config(configs, 'model'),
        'experiment': flatten_config(configs, 'experiment'),
        'prompts': flatten_config(configs, 'prompts')
    }
    
    print(f"Dataset: {full_config['experiment']['dataset_path']}")
    print(f"Images: {full_config['experiment']['num_images']}")
    print(f"Pipelines: {full_config['experiment']['pipelines']}")
    print(f"Repetitions: {full_config['experiment']['repetitions']}")
    print()
    
    # Run experiment
    print("Starting experiment...")
    runner = ExperimentRunner(full_config)
    
    try:
        results = runner.run_experiment()
        print(f"\nCompleted {len(results)} runs")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    main()
