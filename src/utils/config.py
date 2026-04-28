import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file with environment variable substitution"""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace environment variables
    content = os.path.expandvars(content)
    
    return yaml.safe_load(content)


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def load_all_configs(config_dir: str = None) -> Dict[str, Any]:
    """Load all configuration files"""
    if config_dir is None:
        config_dir = get_project_root() / "configs"
    else:
        config_dir = Path(config_dir)
    
    configs = {}
    for config_file in ['model.yaml', 'experiment.yaml', 'prompts.yaml']:
        path = config_dir / config_file
        if path.exists():
            configs[config_file.replace('.yaml', '')] = load_config(str(path))
    
    return configs
