#!/usr/bin/env python3
"""
Granularity sweep experiment for DAM-QA

Tests different window sizes and strides across all datasets to find optimal granularity.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add parent directories to path
sys.path.extend([str(Path(__file__).parent.parent.parent), str(Path(__file__).parent.parent)])

from dam_qa.utils.common import (
    setup_device, load_dam_model, resize_keep_aspect, sliding_window_inference, ExperimentLogger
)
from dam_qa.config import PROMPT_TEMPLATE
from experiments.utils import get_all_dataset_configs, create_experiment_output_path

# Configuration
DEVICE_ID = 1
DETAIL_MODES = {
    "coarse": (768, 384),    # fewer large windows => less detail
    "fine": (256, 128)       # many small windows => more detail
}

def run_granularity_experiment():
    """Run granularity sweep across all datasets."""
    # Setup
    device = setup_device(DEVICE_ID)
    dam_model = load_dam_model(device)
    dataset_configs = get_all_dataset_configs()
    
    logger = ExperimentLogger("Granularity Sweep")
    logger.start()
    
    total_processed = 0
    
    # Iterate through granularity modes and datasets
    for mode, (window_size, stride) in DETAIL_MODES.items():
        logger.log(f"Testing granularity mode: {mode} (window={window_size}, stride={stride})")
        
        for dataset_name, config in dataset_configs.items():
            qa_file = config["qa_file"]
            img_folder = config["img_folder"]
            max_tokens = config["max_new_tokens"]
            
            # Create output path
            output_path = create_experiment_output_path(
                base_dir="experiments/granularity/results",
                experiment_name=f"sliding_{mode}",
                dataset_name=dataset_name,
                split=""
            )
            
            logger.log(f"Processing {dataset_name} -> {output_path}")
            
            # Process dataset
            records = []
            
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
                        
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                    img = resize_keep_aspect(img)
                    
                    # Run sliding window inference
                    prompt = PROMPT_TEMPLATE.format(question=question)
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
                    
                    total_processed += 1
                    
                    if idx % 100 == 0:
                        logger.log(f"Processed {idx} items", idx, idx)
            
            # Save results
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            logger.log(f"Saved {len(records)} rows to {output_path}")
    
    logger.finish(total_processed)


if __name__ == "__main__":
    run_granularity_experiment()
