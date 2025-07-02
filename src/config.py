"""
Configuration file for DAM-QA experiments.

This module contains dataset configurations, prompt templates, and parameters
for running experiments.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


# Base data directory
BASE_DIR = "data"


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    qa_file: str
    img_folder: str
    max_new_tokens: int = 100


class PromptTemplates:
    """Prompt templates for different experiment configurations."""
    
    BASE_TEMPLATE = "Question: {question}\nAnswer:"
    
    # Original prompt template from code_original
    FULL_TEMPLATE = (
        "<image>\n"
        "Answer each question concisely in a single word or short phrase, "
        "without any lengthy descriptions or explanations.\n"
        "Rely only on information that is clearly visible in the provided image.\n"
        "If the answer cannot be determined from the image, respond with \"unanswerable\".\n"
        "Question: {question}\n"
        "Answer:"
    )
    
    VISIBILITY_RULE = (
        "Rely only on information that is clearly visible in the provided image.\n"
    )
    
    UNANSWERABLE_RULE = (
        "If the answer cannot be determined from the image, respond with \"unanswerable\".\n"
    )
    
    @classmethod
    def get_template(cls, use_visibility_rule: bool = True, 
                    use_unanswerable_rule: bool = True) -> str:
        """Get prompt template based on rule settings."""
        template_parts = [
            "<image>\n",
            "Answer each question concisely in a single word or short phrase, "
            "without any lengthy descriptions or explanations.\n"
        ]
        
        if use_visibility_rule:
            template_parts.append(cls.VISIBILITY_RULE)
        
        if use_unanswerable_rule:
            template_parts.append(cls.UNANSWERABLE_RULE)
            
        template_parts.append("Question: {question}\nAnswer:")
        
        return "".join(template_parts)


# Default parameters for model inference (from original code)
DEFAULT_INFERENCE_PARAMS = {
    "streaming": False,
    "temperature": 1e-7,
    "top_p": 0.5,
    "num_beams": 1,
}

# Default parameters for image processing  
DEFAULT_IMAGE_PARAMS = {
    "max_size": 1024,
}

# Default sliding window parameters
DEFAULT_WINDOW_PARAMS = {
    "window_size": 512,
    "stride": 256,
}

# Dataset configurations (flat structure to match VLM config)
DATASET_CONFIGS = {
    "chartqapro_test": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqapro/test.jsonl",
        img_folder=f"{BASE_DIR}/chartqapro/images",
        max_new_tokens=100
    ),
    "chartqa_test_human": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqa/test_human.jsonl",
        img_folder=f"{BASE_DIR}/chartqa/images",
        max_new_tokens=100
    ),
    "chartqa_test_augmented": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqa/test_augmented.jsonl",
        img_folder=f"{BASE_DIR}/chartqa/images",
        max_new_tokens=100
    ),
    "docvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/docvqa/val.jsonl",
        img_folder=f"{BASE_DIR}/docvqa/images",
        max_new_tokens=100
    ),
    "infographicvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/infographicvqa/infographicvqa_val.jsonl",
        img_folder=f"{BASE_DIR}/infographicvqa/images",
        max_new_tokens=100
    ),
    "textvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/textvqa/textvqa_val_updated.jsonl",
        img_folder=f"{BASE_DIR}/textvqa/images",
        max_new_tokens=10
    ),
    "vqav2_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/vqav2/vqav2_restval.jsonl",
        img_folder=f"{BASE_DIR}/vqav2/images",
        max_new_tokens=10
    ),
}

# Granularity sweep parameters (from original code)
GRANULARITY_MODES = {
    "fine": {"window_size": 256, "stride": 128},     # many small windows => more detail
    "medium": {"window_size": 512, "stride": 256},   # default/original
    "coarse": {"window_size": 768, "stride": 384},   # fewer large windows => less detail
}

# Unanswerable weight sweep parameters (from original code)
UNANSWERABLE_WEIGHTS = {
    "low": 0.5,      # 50pct
    "normal": 1.0,   # 100pct  
    "high": 1.5,     # 150pct
    "very_high": 2.0,
}

# Window size and stride sweep parameters
WINDOW_SIZES = [256, 768]
FIXED_STRIDE = 256
STRIDES = [128, 384]
FIXED_WINDOW = 512


def get_dataset_config(dataset_name: str, split: str = "test") -> DatasetConfig:
    """
    Get dataset configuration by name.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (not used currently but kept for compatibility)
        
    Returns:
        DatasetConfig object
        
    Raises:
        ValueError: If dataset_name is not found
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    return DATASET_CONFIGS[dataset_name]


def get_output_path(output_dir: str, experiment_name: str, dataset_name: str, 
                   split: str = "test") -> str:
    """
    Generate output path for experiment results.
    
    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment  
        dataset_name: Dataset name
        split: Dataset split
        
    Returns:
        Full output path for CSV file
    """
    filename = f"{dataset_name}_{split}.csv"
    return os.path.join(output_dir, experiment_name, filename) 