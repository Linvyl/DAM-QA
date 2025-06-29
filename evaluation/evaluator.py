#!/usr/bin/env python3
"""
Main evaluation runner for VQA models.

This script evaluates VLM results using various metrics and provides
comprehensive analysis of model performance.

Usage examples:
    # Evaluate a single result file
    python evaluator.py --file outputs/vlm_results/internvl_chartqapro_test.csv

    # Evaluate all results in a directory
    python evaluator.py --folder outputs/vlm_results/

    # Use LLM-based evaluation
    python evaluator.py --folder outputs/vlm_results/ --use_llm
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import evaluate_folder


def evaluate_single_file(csv_file: str, use_llm: bool = False) -> None:
    """
    Evaluate a single CSV result file.
    
    Args:
        csv_file: Path to CSV file with model results
        use_llm: Whether to use LLM for scoring
    """
    
    folder_path = os.path.dirname(csv_file)
    temp_folder = f"/tmp/single_eval_{os.path.basename(csv_file)}"
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp '{csv_file}' '{temp_folder}/'")
    
    evaluate_folder(temp_folder, use_llm)
    
    results_file = os.path.join(temp_folder, "models_scores.json")
    if os.path.exists(results_file):
        output_file = os.path.join(folder_path, f"{os.path.basename(csv_file)}_scores.json")
        os.system(f"cp '{results_file}' '{output_file}'")
        print(f"Results saved to: {output_file}")
    
    os.system(f"rm -rf '{temp_folder}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model results")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str,
                      help="Single CSV file to evaluate")
    group.add_argument("--folder", type=str,
                      help="Folder containing CSV result files")
    
    parser.add_argument("--use_llm", action="store_true",
                       help="Use LLM for scoring (slower but more accurate)")
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        
        print(f"Evaluating single file: {args.file}")
        evaluate_single_file(args.file, args.use_llm)
        
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return 1
        
        print(f"Evaluating all CSV files in: {args.folder}")
        evaluate_folder(args.folder, args.use_llm)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 