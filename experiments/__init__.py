"""
DAM-QA Experiments Package

This package contains all experimental scripts and utilities for reproducing
the ablation studies and analyses presented in the DAM-QA paper.
"""

from .run_experiments import ExperimentRunner
from .utils import (
    get_all_dataset_configs,
    create_experiment_output_path,
    validate_experimental_setup,
    print_experimental_setup
)

__all__ = [
    "ExperimentRunner",
    "get_all_dataset_configs", 
    "create_experiment_output_path",
    "validate_experimental_setup",
    "print_experimental_setup"
]

__version__ = "1.0.0" 