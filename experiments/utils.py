"""
Experiment utilities for DAM-QA

This module provides utilities for running experiments and accessing
dataset configurations in a unified way.
"""

import os
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATASET_CONFIGS, 
    get_dataset_config, 
    list_available_datasets,
    validate_dataset_paths
)


def get_legacy_config_format(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration in legacy format for experiments.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset configuration with legacy field names
    """
    config = get_dataset_config(dataset_name)
    
    # Convert to legacy format if needed
    if "data_path" in config:
        return {
            "qa_file": config["data_path"],
            "img_folder": config["image_folder"],
            "max_new_tokens": config["max_new_tokens"],
            "metric": config.get("metric", "anls")
        }
    
    # Already in legacy format
    return config


def get_all_dataset_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all dataset configurations for experiments.
    
    Returns:
        Dictionary mapping dataset names to configurations
    """
    return {name: get_legacy_config_format(name) for name in list_available_datasets()}


def create_experiment_output_path(base_dir: str, experiment_name: str, 
                                dataset_name: str, split: str = "",
                                variant: str = "") -> str:
    """
    Create standardized output path for experiments.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        split: Dataset split (if applicable)
        variant: Experiment variant (if applicable)
        
    Returns:
        Full output path
    """
    parts = [base_dir, experiment_name]
    
    if variant:
        parts.append(variant)
    
    # Create filename
    filename_parts = ["dam", dataset_name]
    if split:
        filename_parts.append(split)
    filename = "_".join(filename_parts) + ".csv"
    
    output_dir = os.path.join(*parts)
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, filename)


def validate_experimental_setup() -> Dict[str, bool]:
    """
    Validate that all datasets are properly configured for experiments.
    
    Returns:
        Dictionary mapping dataset names to validation status
    """
    results = {}
    
    for dataset_name in list_available_datasets():
        try:
            validation = validate_dataset_paths(dataset_name)
            results[dataset_name] = (
                validation["data_file_exists"] and 
                validation["image_folder_exists"]
            )
        except Exception as e:
            print(f"Error validating {dataset_name}: {e}")
            results[dataset_name] = False
    
    return results


def print_experimental_setup():
    """Print current experimental setup and dataset status."""
    print("=" * 60)
    print("DAM-QA Experimental Setup")
    print("=" * 60)
    
    validation_results = validate_experimental_setup()
    
    for dataset_name in list_available_datasets():
        status = "✅" if validation_results[dataset_name] else "❌"
        config = get_legacy_config_format(dataset_name)
        
        print(f"\n{status} {dataset_name}")
        print(f"   QA File: {config['qa_file']}")
        print(f"   Images: {config['img_folder']}")
        print(f"   Max tokens: {config['max_new_tokens']}")
        print(f"   Metric: {config.get('metric', 'N/A')}")
    
    print(f"\n{'=' * 60}")
    working = sum(validation_results.values())
    total = len(validation_results)
    print(f"Status: {working}/{total} datasets ready for experiments")
    
    if working < total:
        print("\nTo fix missing datasets:")
        print("1. Check data/datasets/ directory structure") 
        print("2. Update paths in config.py if needed")
        print("3. Run: python data/prepare_datasets.py --base-path /your/data/path")


if __name__ == "__main__":
    print_experimental_setup() 