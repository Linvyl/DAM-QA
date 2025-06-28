#!/usr/bin/env python3
"""
Vote weight sweep experiment for DAM-QA

Tests different weights for "unanswerable" responses across all datasets.
"""

import os
import sys
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel

# Add parent directories to path
sys.path.extend([str(Path(__file__).parent.parent.parent), str(Path(__file__).parent.parent)])

from dam_qa.utils.common import (
    setup_device, load_dam_model, resize_keep_aspect, sliding_window_inference
)
from dam_qa.config import PROMPT_TEMPLATE
from experiments.utils import get_all_dataset_configs, create_experiment_output_path

# -------- Global Configurations --------
DEVICE_ID = 2

VOTE_WEIGHTS = {
    "50pct": 0.5,
    "100pct": 1.0,
    "150pct": 1.5
}

WINDOW_SIZE = 512
STRIDE = 256


def run_vote_weight_sweep():
    """Run vote weight sweep across all datasets."""
    # Setup
    device = setup_device(DEVICE_ID)
    dam_model = load_dam_model(device)
    dataset_configs = get_all_dataset_configs()
    
    print(f"Using device: {device}")
    
    # Execute sweep over unanswerable weights
    for tag, unanswer_weight in VOTE_WEIGHTS.items():
        print(f"\n=== Sweep UNANSWER_WEIGHT={unanswer_weight} ({tag}) ===")
        
        for ds_name, cfg in dataset_configs.items():
            qa_file = cfg["qa_file"]
            img_folder = cfg["img_folder"]
            max_tokens = cfg["max_new_tokens"]
            
            # Create output path
            output_path = create_experiment_output_path(
                base_dir="experiments/vote_weights/results",
                experiment_name=f"weight_{tag}",
                dataset_name=ds_name
            )
            
            # Skip if file exists
            if os.path.exists(output_path):
                print(f"→ {ds_name}: {output_path} exists, skip.")
                continue

            # Count total lines
            with open(qa_file, "r") as f:
                total = sum(1 for _ in f)
            
            records = []
            t0 = time.time()
            
            with open(qa_file, "r") as fin:
                for line in tqdm(fin, total=total, desc=f"{ds_name}"):
                    entry = json.loads(line)
                    qid = entry.get("question_id", None)
                    question = entry.get("question", "").strip()
                    img_name = entry.get("image") or entry.get("image_id")
                    gt = entry.get("answer", [])

                    img_path = os.path.join(img_folder, img_name)
                    if not os.path.exists(img_path):
                        continue

                    # Load and process image
                    img = Image.open(img_path).convert("RGB")
                    img = resize_keep_aspect(img)
                    
                    # Run sliding window inference with vote weighting
                    prompt = PROMPT_TEMPLATE.format(question=question)
                    prediction = sliding_window_inference(
                        dam_model=dam_model,
                        image=img,
                        prompt=prompt,
                        window_size=WINDOW_SIZE,
                        stride=STRIDE,
                        max_new_tokens=max_tokens,
                        unanswerable_weight=unanswer_weight
                    )
                    
                    records.append({
                        "question_id": qid,
                        "image_id": img_name,
                        "question": question,
                        "predict": prediction,
                        "gt": gt
                    })
            
            # Save results
            pd.DataFrame(records).to_csv(output_path, index=False)
            elapsed = (time.time() - t0) / 60
            print(f"→ {ds_name}: saved {len(records)} rows in {elapsed:.2f} min")


if __name__ == "__main__":
    run_vote_weight_sweep()

