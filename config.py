"""
Unified Configuration for DAM-QA Project

This module contains all configurations for the DAM-QA project including:
- Model parameters and hyperparameters  
- Dataset configurations for all supported datasets
- Path configurations with environment-based customization
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DAMConfig:
    """Configuration class for DAM-QA model."""
    
    # Model parameters
    model_name: str = "nvidia/DAM-3B-Self-Contained"
    conv_mode: str = "v1"
    prompt_mode: str = "full+focal_crop"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    
    # Sliding window parameters
    window_size: int = 512
    stride: int = 256
    max_image_size: int = 1024
    
    # Generation parameters
    temperature: float = 1e-7
    top_p: float = 0.5
    num_beams: int = 1
    max_new_tokens: int = 100
    
    # Vote weighting
    unanswerable_weight: float = 1.0  # Weight for "unanswerable" responses
    
    # Device settings
    device_id: int = 0


# Base dataset directory - can be overridden by environment variable
BASE_DATA_DIR = os.environ.get("DAM_DATA_DIR", "data/datasets")


def get_dataset_path(relative_path: str) -> str:
    """Get absolute path for dataset files."""
    return os.path.join(BASE_DATA_DIR, relative_path)


# Unified dataset configurations
# Used by both main DAM-QA inference and evaluation framework
DATASET_CONFIGS = {
    "infographicvqa_val": {
        "data_path": get_dataset_path("infographicvqa/infographicvqa_val.jsonl"),
        "image_folder": get_dataset_path("infographicvqa/images"),
        "max_new_tokens": 100,
        "metric": "anls",
        "description": "InfographicVQA validation set"
    },
    
    "docvqa_val": {
        "data_path": get_dataset_path("docvqa/val.jsonl"),
        "image_folder": get_dataset_path("docvqa/images"),
        "max_new_tokens": 100,
        "metric": "anls",
        "description": "DocVQA validation set"
    },
    
    "chartqa_test_human": {
        "data_path": get_dataset_path("chartqa/test_human.jsonl"),
        "image_folder": get_dataset_path("chartqa/images"),
        "max_new_tokens": 100,
        "metric": "relaxed_accuracy",
        "description": "ChartQA human test set"
    },
    
    "chartqa_test_augmented": {
        "data_path": get_dataset_path("chartqa/test_augmented.jsonl"),
        "image_folder": get_dataset_path("chartqa/images"),
        "max_new_tokens": 100,
        "metric": "relaxed_accuracy",
        "description": "ChartQA augmented test set"
    },
    
    "chartqapro_test": {
        "data_path": get_dataset_path("chartqapro/test.jsonl"),
        "image_folder": get_dataset_path("chartqapro/images"),
        "max_new_tokens": 100,
        "metric": "chartqapro",
        "description": "ChartQA-Pro test set"
    },
    
    "textvqa_val": {
        "data_path": get_dataset_path("textvqa/textvqa_val_updated.jsonl"),
        "image_folder": get_dataset_path("textvqa/images"),
        "max_new_tokens": 10,
        "metric": "vqa_score",
        "description": "TextVQA validation set"
    },
    
    "vqav2_restval": {
        "data_path": get_dataset_path("vqav2/vqav2_restval.jsonl"),
        "image_folder": get_dataset_path("vqav2/images"),
        "max_new_tokens": 10,
        "metric": "vqa_score",
        "description": "VQAv2 restval set"
    }
}

# Legacy aliases for backward compatibility
LEGACY_DATASET_CONFIGS = {
    dataset_name: {
        "qa_file": config["data_path"],
        "img_folder": config["image_folder"],
        "max_new_tokens": config["max_new_tokens"],
        "metric": config["metric"]
    }
    for dataset_name, config in DATASET_CONFIGS.items()
}

# Evaluation framework compatibility
ds_collections = DATASET_CONFIGS.copy()

# Prompt templates
PROMPT_TEMPLATE = (
    "<image>\n"
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with \"unanswerable\".\n"
    "Question: {question}\n"
    "Answer:"
)

# Prompt variations for experiments
PROMPT_VARIANTS = {
    "full": PROMPT_TEMPLATE,
    "no_rules": (
        "<image>\n"
        "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "rule1_only": (
        "<image>\n"
        "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
        "Rely only on information that is clearly visible in the provided image.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "rule2_only": (
        "<image>\n"
        "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
        "If the answer cannot be determined from the image, respond with \"unanswerable\".\n"
        "Question: {question}\n"
        "Answer:"
    )
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration by name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset configuration dictionary
        
    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def list_available_datasets() -> list:
    """List all available dataset names."""
    return list(DATASET_CONFIGS.keys())


def validate_dataset_paths(dataset_name: str) -> Dict[str, bool]:
    """
    Validate that dataset paths exist.
    
    Args:
        dataset_name: Name of the dataset to validate
        
    Returns:
        Dictionary with validation results
    """
    config = get_dataset_config(dataset_name)
    
    return {
        "data_file_exists": os.path.exists(config["data_path"]),
        "image_folder_exists": os.path.exists(config["image_folder"]),
        "data_path": config["data_path"],
        "image_folder": config["image_folder"]
    }


# Experimental configurations for ablation studies
GRANULARITY_CONFIGS = {
    "coarse": {"window_size": 768, "stride": 384},
    "default": {"window_size": 512, "stride": 256}, 
    "fine": {"window_size": 256, "stride": 128}
}

VOTE_WEIGHT_CONFIGS = {
    "equal": 1.0,
    "slight_boost": 1.2,
    "medium_boost": 1.5,
    "strong_boost": 2.0
} 