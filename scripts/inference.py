#!/usr/bin/env python3
"""
DAM-QA Inference Script

Run inference using either the sliding window approach or baseline full-image method
on various VQA datasets.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from dam_qa import DAMSlidingWindow, DAMBaseline
from config import DATASET_CONFIGS, DAMConfig


def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.
    
    Args:
        dataset_name: Name of the dataset configuration
        
    Returns:
        List of data items
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in configuration")
    
    config = DATASET_CONFIGS[dataset_name]
    qa_file = config.get("qa_file") or config.get("data_path")
    img_folder = config.get("img_folder") or config.get("image_folder")
    
    if not qa_file:
        raise ValueError(f"No qa_file or data_path found for dataset {dataset_name}")
    if not img_folder:
        raise ValueError(f"No img_folder or image_folder found for dataset {dataset_name}")
    
    data_items = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            
            # Prepare data item
            image_name = item.get("image") or item.get("image_id")
            data_item = {
                "question_id": item.get("question_id"),
                "image_id": image_name,
                "image_path": os.path.join(img_folder, image_name),
                "question": item.get("question", "").strip(),
                "gt": item.get("answer", [])
            }
            data_items.append(data_item)
    
    return data_items


def main():
    parser = argparse.ArgumentParser(description="DAM-QA Inference Script")
    parser.add_argument("--method", choices=["sliding", "baseline"], required=True,
                        help="Inference method to use")
    parser.add_argument("--dataset", required=True, 
                        help="Dataset name (e.g., infographicvqa_val)")
    parser.add_argument("--output-dir", default="./results",
                        help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--window-size", type=int, default=512,
                        help="Sliding window size")
    parser.add_argument("--stride", type=int, default=256,
                        help="Sliding window stride")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum new tokens to generate")
    parser.add_argument("--unanswerable-weight", type=float, default=1.0,
                        help="Weight for unanswerable responses")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data_items = load_dataset(args.dataset)
    print(f"Loaded {len(data_items)} items")
    
    # Create configuration
    config = DAMConfig(
        device_id=args.gpu,
        window_size=args.window_size,
        stride=args.stride,
        unanswerable_weight=args.unanswerable_weight
    )
    
    # Override max_tokens if specified
    if args.max_tokens is not None:
        config.max_new_tokens = args.max_tokens
    elif args.dataset in DATASET_CONFIGS:
        config.max_new_tokens = DATASET_CONFIGS[args.dataset]["max_new_tokens"]
    
    # Initialize model
    if args.method == "sliding":
        print("Using DAM Sliding Window approach")
        model = DAMSlidingWindow(config)
    else:
        print("Using DAM Baseline approach")
        model = DAMBaseline(config)
    
    # Create output directory and path
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, 
        f"dam_{args.method}_{args.dataset}.csv"
    )
    
    # Run inference
    print(f"Starting inference with {args.method} method...")
    results = model.batch_inference(data_items, output_path)
    
    print(f"Inference completed! Results saved to {output_path}")
    print(f"Processed {len(results)} items")


if __name__ == "__main__":
    main() 