"""VQA Dataset Configurations

This module contains dataset configurations for various VQA benchmarks
including paths, prompts, metrics, and generation parameters.
"""
BASE_DIR = "/mnt/VLAI_data"

ds_collections = {
    "vqav2_val": {
        "data_path": f"{BASE_DIR}/VQAv2/vqav2_restval.jsonl",
        "image_folder": f"{BASE_DIR}/COCO_Images/val2014/",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    
    "vqav2_restval": {
        "data_path": f"{BASE_DIR}/VQAv2/vqav2_restval.jsonl",
        "image_folder": f"{BASE_DIR}/COCO_Images/val2014/",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    
    "infographicvqa_val": {
        "data_path": f"{BASE_DIR}/InfographicVQA/infographicvqa_val.jsonl",
        "image_folder": f"{BASE_DIR}/InfographicVQA/images",
        "metric": "anls",
        "max_new_tokens": 100,
    },
    
    "textvqa_val": {
        "data_path": f"{BASE_DIR}/TextVQA/textvqa_val_updated.jsonl",
        "image_folder": f"{BASE_DIR}/TextVQA/train_images",
        "metric": "vqa_score",
        "max_new_tokens": 10,
    },
    
    "chartqa_test_human": {
        "data_path": f"{BASE_DIR}/ChartQA/test_human.jsonl",
        "image_folder": f"{BASE_DIR}/ChartQA",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    
    "chartqa_test_augmented": {
        "data_path": f"{BASE_DIR}/ChartQA/test_augmented.jsonl",
        "image_folder": f"{BASE_DIR}/ChartQA",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    
    "chartqapro_test": {
        "data_path": f"{BASE_DIR}/ChartQAPro/test.jsonl",
        "image_folder": f"{BASE_DIR}/ChartQAPro/images",
        "metric": "chartqapro",
        "max_new_tokens": 100,
    },
    
    "docvqa_val": {
        "data_path": f"{BASE_DIR}/DocVQA/val.jsonl",
        "image_folder": f"{BASE_DIR}/DocVQA/images",
        "metric": "anls",
        "max_new_tokens": 100,
    },
}
