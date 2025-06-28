"""VQA Dataset Configurations

This module imports dataset configurations from the unified config.py and provides
backward compatibility for the evaluation framework.
"""

import os
import sys

# Add parent directory to path to import unified config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config import DATASET_CONFIGS as unified_configs
    
    # Use unified config directly - already in correct format
    ds_collections = unified_configs.copy()
        
except ImportError:
    # Fallback to local definitions if unified config not available
    BASE_DIR = "../data/datasets"
    
    ds_collections = {    
        "vqav2_restval": {
            "data_path": f"{BASE_DIR}/vqav2/vqav2_restval.jsonl",
            "image_folder": f"{BASE_DIR}/vqav2/images/",
            "metric": "vqa_score",
            "max_new_tokens": 10,
        },
        
        "infographicvqa_val": {
            "data_path": f"{BASE_DIR}/infographicvqa/infographicvqa_val.jsonl",
            "image_folder": f"{BASE_DIR}/infographicvqa/images",
            "metric": "anls",
            "max_new_tokens": 100,
        },
        
        "textvqa_val": {
            "data_path": f"{BASE_DIR}/textvqa/textvqa_val_updated.jsonl",
            "image_folder": f"{BASE_DIR}/textvqa/images",
            "metric": "vqa_score",
            "max_new_tokens": 10,
        },
        
        "chartqa_test_human": {
            "data_path": f"{BASE_DIR}/chartqa/test_human.jsonl",
            "image_folder": f"{BASE_DIR}/chartqa/images",
            "metric": "relaxed_accuracy",
            "max_new_tokens": 100,
        },
        
        "chartqa_test_augmented": {
            "data_path": f"{BASE_DIR}/chartqa/test_augmented.jsonl",
            "image_folder": f"{BASE_DIR}/chartqa/images",
            "metric": "relaxed_accuracy",
            "max_new_tokens": 100,
        },
        
        "chartqapro_test": {
            "data_path": f"{BASE_DIR}/chartqapro/test.jsonl",
            "image_folder": f"{BASE_DIR}/chartqapro/images",
            "metric": "chartqapro",
            "max_new_tokens": 100,
        },
        
        "docvqa_val": {
            "data_path": f"{BASE_DIR}/docvqa/val.jsonl",
            "image_folder": f"{BASE_DIR}/docvqa/images",
            "metric": "anls",
            "max_new_tokens": 100,
        },
    }
