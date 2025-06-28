#!/usr/bin/env python3
"""
Prompt design experiment: No rules

Tests sliding window inference without any prompt rules.
Note: This is a legacy script. Use run_experiments.py for new experiments.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.extend([str(Path(__file__).parent.parent.parent), str(Path(__file__).parent.parent)])

from dam_qa.utils.common import (
    setup_device, load_dam_model, sliding_window_inference, resize_keep_aspect
)
from dam_qa.config import PROMPT_VARIANTS
from experiments.utils import get_all_dataset_configs, create_experiment_output_path

import json
import time
import pandas as pd
from PIL import Image


def run_no_rule_experiment():
    """Run prompt experiment with no rules."""
    # Setup
    device = setup_device(0)  # Use GPU 0
    dam_model = load_dam_model(device)
    dataset_configs = get_all_dataset_configs()
    
    # Experiment configuration
    window_size = 512
    stride = 256
    prompt_template = PROMPT_VARIANTS["no_rules"]
    
    print("Running sliding window experiment without rules...")
    
    for ds_name, config in dataset_configs.items():
        qa_file = config["qa_file"]
        img_folder = config["img_folder"]
        max_tokens = config["max_new_tokens"]
        
        # Create output path
        output_path = create_experiment_output_path(
            base_dir="experiments/prompt_design/results",
            experiment_name="sliding_no_rules",
            dataset_name=ds_name
        )
        
        print(f"Processing {ds_name} -> {output_path}")
        
        records = []
        start_time = time.time()
        
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
                    max_new_tokens=max_tokens
                )
                
                records.append({
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt
                })
                
                if idx % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {idx} items in {elapsed:.1f}s")
        
        # Save results
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        total_time = (time.time() - start_time) / 60
        print(f"Saved {len(records)} rows in {total_time:.2f} min")


if __name__ == "__main__":
    print("=" * 60)
    print("DEPRECATION WARNING:")
    print("This individual script is deprecated. Please use:")
    print("python experiments/run_experiments.py --experiment prompt --method sliding")
    print("=" * 60)
    
    run_no_rule_experiment()
