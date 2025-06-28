#!/usr/bin/env python3
"""
Unified experiment runner for DAM-QA

This script provides a unified interface to run different types of experiments
using the optimized utilities and avoiding code duplication.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dam_qa.utils.common import (
    setup_device, load_dam_model, sliding_window_inference, 
    resize_keep_aspect, ExperimentLogger
)
from dam_qa.config import PROMPT_TEMPLATE, PROMPT_VARIANTS
from experiments.utils import (
    get_all_dataset_configs, create_experiment_output_path,
    print_experimental_setup
)


class ExperimentRunner:
    """Unified experiment runner for DAM-QA."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize experiment runner.
        
        Args:
            device_id: CUDA device ID to use
        """
        self.device = setup_device(device_id)
        self.dam_model = None
        self.dataset_configs = get_all_dataset_configs()
        
    def load_model(self):
        """Load DAM model once for all experiments."""
        if self.dam_model is None:
            self.dam_model = load_dam_model(self.device)
        return self.dam_model
    
    def run_granularity_experiment(self, modes: Dict[str, tuple], 
                                 datasets: Optional[List[str]] = None) -> None:
        """
        Run granularity sweep experiment.
        
        Args:
            modes: Dictionary mapping mode names to (window_size, stride) tuples
            datasets: List of dataset names to test (None for all)
        """
        dam_model = self.load_model()
        datasets = datasets or list(self.dataset_configs.keys())
        
        logger = ExperimentLogger("Granularity Sweep")
        logger.start()
        
        total_processed = 0
        
        for mode, (window_size, stride) in modes.items():
            logger.log(f"Testing mode: {mode} (window={window_size}, stride={stride})")
            
            for dataset_name in datasets:
                if dataset_name not in self.dataset_configs:
                    logger.log(f"Dataset {dataset_name} not found, skipping")
                    continue
                
                total_processed += self._process_dataset(
                    dam_model=dam_model,
                    dataset_name=dataset_name,
                    experiment_name=f"granularity/sliding_{mode}",
                    window_size=window_size,
                    stride=stride,
                    prompt_template=PROMPT_TEMPLATE,
                    logger=logger
                )
        
        logger.finish(total_processed)
    
    def run_prompt_experiment(self, prompt_variants: Dict[str, str],
                            datasets: Optional[List[str]] = None,
                            method: str = "sliding") -> None:
        """
        Run prompt design experiment.
        
        Args:
            prompt_variants: Dictionary mapping variant names to prompt templates
            datasets: List of dataset names to test (None for all)
            method: "sliding" or "baseline"
        """
        dam_model = self.load_model()
        datasets = datasets or list(self.dataset_configs.keys())
        
        logger = ExperimentLogger(f"Prompt Design ({method})")
        logger.start()
        
        total_processed = 0
        
        for variant_name, prompt_template in prompt_variants.items():
            logger.log(f"Testing prompt variant: {variant_name}")
            
            for dataset_name in datasets:
                if dataset_name not in self.dataset_configs:
                    continue
                
                if method == "sliding":
                    total_processed += self._process_dataset(
                        dam_model=dam_model,
                        dataset_name=dataset_name,
                        experiment_name=f"prompt_design/sliding_{variant_name}",
                        window_size=512,
                        stride=256,
                        prompt_template=prompt_template,
                        logger=logger
                    )
                else:  # baseline
                    total_processed += self._process_dataset_baseline(
                        dam_model=dam_model,
                        dataset_name=dataset_name,
                        experiment_name=f"prompt_design/baseline_{variant_name}",
                        prompt_template=prompt_template,
                        logger=logger
                    )
        
        logger.finish(total_processed)
    
    def run_vote_weight_experiment(self, weight_configs: Dict[str, float],
                                 datasets: Optional[List[str]] = None) -> None:
        """
        Run vote weighting experiment.
        
        Args:
            weight_configs: Dictionary mapping weight names to multiplier values
            datasets: List of dataset names to test (None for all)
        """
        dam_model = self.load_model()
        datasets = datasets or list(self.dataset_configs.keys())
        
        logger = ExperimentLogger("Vote Weight Sweep")
        logger.start()
        
        total_processed = 0
        
        for weight_name, unanswerable_weight in weight_configs.items():
            logger.log(f"Testing vote weight: {weight_name} (weight={unanswerable_weight})")
            
            for dataset_name in datasets:
                if dataset_name not in self.dataset_configs:
                    continue
                
                total_processed += self._process_dataset(
                    dam_model=dam_model,
                    dataset_name=dataset_name,
                    experiment_name=f"vote_weights/weight_{weight_name}",
                    window_size=512,
                    stride=256,
                    prompt_template=PROMPT_TEMPLATE,
                    unanswerable_weight=unanswerable_weight,
                    logger=logger
                )
        
        logger.finish(total_processed)
    
    def _process_dataset(self, dam_model, dataset_name: str, experiment_name: str,
                        window_size: int, stride: int, prompt_template: str,
                        unanswerable_weight: float = 1.0,
                        logger: Optional[ExperimentLogger] = None) -> int:
        """Process a single dataset with sliding window method."""
        import json
        import pandas as pd
        from PIL import Image
        import os
        
        config = self.dataset_configs[dataset_name]
        qa_file = config["qa_file"]
        img_folder = config["img_folder"]
        max_tokens = config["max_new_tokens"]
        
        # Create output path
        output_path = create_experiment_output_path(
            base_dir="experiments/results",
            experiment_name=experiment_name,
            dataset_name=dataset_name
        )
        
        if logger:
            logger.log(f"Processing {dataset_name} -> {output_path}")
        
        records = []
        processed_count = 0
        
        with open(qa_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin, 1):
                entry = json.loads(line)
                qid = entry.get("question_id", idx)
                question = entry["question"].strip()
                img_name = entry.get("image") or entry.get("image_id")
                gt = entry.get("answer", [])
                
                # Load and process image
                img_path = os.path.join(img_folder, img_name)
                if not os.path.exists(img_path):
                    continue
                
                img = Image.open(img_path).convert("RGB")
                img = resize_keep_aspect(img)
                
                # Run inference
                prompt = prompt_template.format(question=question)
                prediction = sliding_window_inference(
                    dam_model=dam_model,
                    image=img,
                    prompt=prompt,
                    window_size=window_size,
                    stride=stride,
                    max_new_tokens=max_tokens,
                    unanswerable_weight=unanswerable_weight
                )
                
                records.append({
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt
                })
                
                processed_count += 1
                
                if idx % 100 == 0 and logger:
                    logger.log(f"Processed {idx} items from {dataset_name}")
        
        # Save results
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        if logger:
            logger.log(f"Saved {len(records)} rows to {output_path}")
        
        return processed_count
    
    def _process_dataset_baseline(self, dam_model, dataset_name: str, 
                                experiment_name: str, prompt_template: str,
                                logger: Optional[ExperimentLogger] = None) -> int:
        """Process a single dataset with baseline (full-image) method."""
        import json
        import pandas as pd
        from PIL import Image
        import os
        from dam_qa.utils.common import create_full_mask, dam_inference
        
        config = self.dataset_configs[dataset_name]
        qa_file = config["qa_file"]
        img_folder = config["img_folder"]
        max_tokens = config["max_new_tokens"]
        
        # Create output path
        output_path = create_experiment_output_path(
            base_dir="experiments/results",
            experiment_name=experiment_name,
            dataset_name=dataset_name
        )
        
        if logger:
            logger.log(f"Processing {dataset_name} (baseline) -> {output_path}")
        
        records = []
        processed_count = 0
        
        with open(qa_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin, 1):
                entry = json.loads(line)
                qid = entry.get("question_id", idx)
                question = entry["question"].strip()
                img_name = entry.get("image") or entry.get("image_id")
                gt = entry.get("answer", [])
                
                # Load and process image
                img_path = os.path.join(img_folder, img_name)
                if not os.path.exists(img_path):
                    continue
                
                img = Image.open(img_path).convert("RGB")
                img = resize_keep_aspect(img)
                
                # Run full-image inference
                W, H = img.size
                mask = create_full_mask(W, H)
                prompt = prompt_template.format(question=question)
                prediction = dam_inference(
                    dam_model=dam_model,
                    image=img,
                    mask=mask,
                    prompt=prompt,
                    max_new_tokens=max_tokens
                )
                
                records.append({
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt
                })
                
                processed_count += 1
                
                if idx % 100 == 0 and logger:
                    logger.log(f"Processed {idx} items from {dataset_name}")
        
        # Save results
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        if logger:
            logger.log(f"Saved {len(records)} rows to {output_path}")
        
        return processed_count


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="DAM-QA Experiment Runner")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["granularity", "prompt", "vote_weights", "setup"],
                        help="Type of experiment to run")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument("--datasets", type=str, nargs="+",
                        help="Specific datasets to test (default: all)")
    parser.add_argument("--method", type=str, choices=["sliding", "baseline"], 
                        default="sliding",
                        help="Method to use for prompt experiments")
    
    args = parser.parse_args()
    
    if args.experiment == "setup":
        print_experimental_setup()
        return
    
    # Initialize runner
    runner = ExperimentRunner(device_id=args.device)
    
    if args.experiment == "granularity":
        modes = {
            "coarse": (768, 384),
            "fine": (256, 128)
        }
        runner.run_granularity_experiment(modes, args.datasets)
    
    elif args.experiment == "prompt":
        prompt_variants = {
            "full": PROMPT_VARIANTS["full"],
            "no_rules": PROMPT_VARIANTS["no_rules"],
            "rule1_only": PROMPT_VARIANTS["rule1_only"],
            "rule2_only": PROMPT_VARIANTS["rule2_only"]
        }
        runner.run_prompt_experiment(prompt_variants, args.datasets, args.method)
    
    elif args.experiment == "vote_weights":
        weight_configs = {
            "equal": 1.0,
            "slight_boost": 1.2,
            "medium_boost": 1.5,
            "strong_boost": 2.0
        }
        runner.run_vote_weight_experiment(weight_configs, args.datasets)


if __name__ == "__main__":
    main() 