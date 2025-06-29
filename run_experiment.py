#!/usr/bin/env python3
"""
Main script to run DAM-QA experiments.

This script provides a unified interface to run various experiments including:
- Full image inference (baseline)
- Sliding window inference with different parameters
- Ablation studies for prompt design, granularity, and vote weights

Usage examples:
    # Full image baseline
    python run_experiment.py --method full_image --dataset chartqapro_test --gpu 0

    # Sliding window inference
    python run_experiment.py --method sliding_window --dataset chartqapro_test --gpu 0

    # Granularity sweep
    python run_experiment.py --method granularity_sweep --dataset chartqapro_test --gpu 0

    # Prompt design ablation
    python run_experiment.py --method prompt_ablation --dataset chartqapro_test --gpu 0

    # Run on all datasets
    python run_experiment.py --method sliding_window --dataset all --gpu 0
"""

import argparse
import sys
import os
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core import FullImageInference, SlidingWindowInference, run_experiment
from src.config import (
    DATASET_CONFIGS, get_dataset_config, get_output_path,
    GRANULARITY_MODES, UNANSWERABLE_WEIGHTS
)


def run_single_experiment(args):
    """Run a single experiment configuration."""
    print(f"Running {args.method} on {args.dataset}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataset_config = get_dataset_config(args.dataset)
    
    experiment_suffix = args.experiment_suffix or "default"
    output_path = get_output_path(args.output_dir, f"{args.method}_{experiment_suffix}", 
                                 args.dataset)
    
    # Initialize and run inference
    if args.method == "full_image":
        inference = FullImageInference(device="auto")
        inference.run_dataset(
            dataset_config, output_path,
            use_visibility_rule=args.use_visibility_rule,
            use_unanswerable_rule=args.use_unanswerable_rule
        )
    
    elif args.method == "sliding_window":
        inference = SlidingWindowInference(
            device="auto",
            window_size=args.window_size,
            stride=args.stride
        )
        inference.run_dataset(
            dataset_config, output_path,
            use_visibility_rule=args.use_visibility_rule,
            use_unanswerable_rule=args.use_unanswerable_rule,
            unanswerable_weight=args.unanswerable_weight
        )
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    print(f"Results saved to: {output_path}")


def run_granularity_sweep(args):
    """Run granularity parameter sweep."""
    print("Running granularity sweep...")
    
    # Store original method
    original_method = args.method
    
    for mode, params in GRANULARITY_MODES.items():
        print(f"\n=== Granularity mode: {mode} ===")
        
        # Override window parameters
        args.window_size = params["window_size"]
        args.stride = params["stride"]
        args.experiment_suffix = f"granularity_{mode}"
        args.method = "sliding_window"  # Use sliding window for granularity sweep
        
        if args.dataset == "all":
            run_all_datasets(args)
        else:
            run_single_experiment(args)
    
    # Restore original method
    args.method = original_method


def run_unanswerable_weight_sweep(args):
    """Run unanswerable vote weight sweep."""
    print("Running unanswerable weight sweep...")
    
    # Store original method
    original_method = args.method
    
    for tag, weight in UNANSWERABLE_WEIGHTS.items():
        print(f"\n=== Unanswerable weight: {weight} ({tag}) ===")
        
        # Override weight parameter
        args.unanswerable_weight = weight
        args.experiment_suffix = f"unans_weight_{tag}"
        args.method = "sliding_window"  # Use sliding window for weight sweep
        
        if args.dataset == "all":
            run_all_datasets(args)
        else:
            run_single_experiment(args)
    
    # Restore original method
    args.method = original_method


def run_prompt_ablation(args):
    """Run prompt design ablation study."""
    print("Running prompt design ablation...")
    
    # Store original method
    original_method = args.method
    
    prompt_configs = [
        {"use_visibility_rule": False, "use_unanswerable_rule": False, "suffix": "no_rules"},
        {"use_visibility_rule": True, "use_unanswerable_rule": False, "suffix": "visibility_only"},
        {"use_visibility_rule": False, "use_unanswerable_rule": True, "suffix": "unanswerable_only"},
        {"use_visibility_rule": True, "use_unanswerable_rule": True, "suffix": "full_rules"}
    ]
    
    for config in prompt_configs:
        print(f"\n=== Prompt config: {config['suffix']} ===")
        
        # Override prompt parameters
        args.use_visibility_rule = config["use_visibility_rule"]
        args.use_unanswerable_rule = config["use_unanswerable_rule"]
        args.experiment_suffix = f"prompt_{config['suffix']}"
        args.method = "sliding_window"  # Use sliding window for prompt ablation
        
        if args.dataset == "all":
            run_all_datasets(args)
        else:
            run_single_experiment(args)
    
    # Restore original method
    args.method = original_method


def run_window_size_sweep(args):
    """Run window size parameter sweep."""
    print("Running window size sweep...")
    
    # Store original method
    original_method = args.method
    
    window_sizes = [256, 768]  # Keep stride fixed at 256
    fixed_stride = 256
    
    for ws in window_sizes:
        print(f"\n=== Window size: {ws} ===")
        
        args.window_size = ws
        args.stride = fixed_stride
        args.experiment_suffix = f"ws_sweep_ws{ws}_st{fixed_stride}"
        args.method = "sliding_window"  # Use sliding window for window size sweep
        
        if args.dataset == "all":
            run_all_datasets(args)
        else:
            run_single_experiment(args)
    
    # Restore original method
    args.method = original_method


def run_stride_sweep(args):
    """Run stride parameter sweep."""
    print("Running stride sweep...")
    
    # Store original method
    original_method = args.method
    
    strides = [128, 384]  # Keep window size fixed at 512
    fixed_window = 512
    
    for stride in strides:
        print(f"\n=== Stride: {stride} ===")
        
        args.window_size = fixed_window
        args.stride = stride
        args.experiment_suffix = f"stride_sweep_ws{fixed_window}_st{stride}"
        args.method = "sliding_window"  # Use sliding window for stride sweep
        
        if args.dataset == "all":
            run_all_datasets(args)
        else:
            run_single_experiment(args)
    
    # Restore original method
    args.method = original_method


def run_all_datasets(args):
    """Run experiment on all dataset configurations."""
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"\n--- Running {dataset_name} ---")
        original_dataset = args.dataset        
        args.dataset = dataset_name
        
        try:
            run_single_experiment(args)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
        
        # Restore original values
        args.dataset = original_dataset


def main():
    parser = argparse.ArgumentParser(description="Run DAM-QA experiments")
    
    # Core arguments
    parser.add_argument("--method", type=str, required=True,
                       choices=["full_image", "sliding_window", "granularity_sweep", 
                               "unanswerable_weight_sweep", "prompt_ablation",
                               "window_size_sweep", "stride_sweep"],
                       help="Experiment method to run")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()) + ["all"],
                       help="Dataset to use ('all' for all datasets)")
    
    parser.add_argument("--gpu", type=str, default="0",
                       help="GPU device ID(s)")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")
    
    # Method-specific arguments
    parser.add_argument("--window_size", type=int, default=512,
                       help="Sliding window size")
    
    parser.add_argument("--stride", type=int, default=256,
                       help="Sliding window stride")
    
    parser.add_argument("--unanswerable_weight", type=float, default=1.0,
                       help="Weight multiplier for unanswerable votes")
    
    # Prompt design arguments
    parser.add_argument("--use_visibility_rule", action="store_true", default=True,
                       help="Include visibility constraint in prompt")
    
    parser.add_argument("--use_unanswerable_rule", action="store_true", default=True,
                       help="Include unanswerable instruction in prompt")
    
    parser.add_argument("--no_visibility_rule", action="store_true",
                       help="Disable visibility constraint")
    
    parser.add_argument("--no_unanswerable_rule", action="store_true",
                       help="Disable unanswerable instruction")
    
    # Other arguments
    parser.add_argument("--experiment_suffix", type=str,
                       help="Suffix for experiment name")
    
    args = parser.parse_args()
    
    # Handle negative flags
    if args.no_visibility_rule:
        args.use_visibility_rule = False
    if args.no_unanswerable_rule:
        args.use_unanswerable_rule = False
    
    # Validate dataset combination
    if args.dataset != "all":
        try:
            get_dataset_config(args.dataset)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Run the appropriate experiment
    try:
        if args.method in ["full_image", "sliding_window"]:
            if args.dataset == "all":
                run_all_datasets(args)
            else:
                run_single_experiment(args)
        
        elif args.method == "granularity_sweep":
            run_granularity_sweep(args)
        
        elif args.method == "unanswerable_weight_sweep":
            run_unanswerable_weight_sweep(args)
        
        elif args.method == "prompt_ablation":
            run_prompt_ablation(args)
            
        elif args.method == "window_size_sweep":
            run_window_size_sweep(args)
            
        elif args.method == "stride_sweep":
            run_stride_sweep(args)
        
        print("\nExperiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 