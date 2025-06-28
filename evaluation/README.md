# VQA Evaluation Framework

A comprehensive evaluation framework for Visual Question Answering (VQA) models, integrated with the DAM-QA project. This framework enables systematic comparison of state-of-the-art VQA models across multiple benchmarks.

## Overview

This framework provides a unified interface for evaluating various state-of-the-art VQA models on popular benchmarks including VQAv2, TextVQA, ChartQA, DocVQA, and InfographicVQA. It's specifically designed to work with the DAM-QA sliding window approach and can be used to compare against other baseline models.

## Features

- **Multi-Model Support**: Evaluate 7+ different VQA models including QwenVL, InternVL, Molmo, VideoLLaMA, MiniCPM, Ovis, and Phi
- **Multi-Dataset Support**: Support for 6 major VQA benchmarks used in DAM-QA
- **Batch Processing**: Efficient batch inference for faster evaluation
- **Configurable Output**: Customizable output directories and formats
- **Type-Safe Code**: Comprehensive type hints for better code quality
- **Modular Design**: Easy to extend with new models and datasets
- **Environment Variable Configuration**: Flexible model path configuration

## Supported Models

| Model | Description | Environment Variable |
|-------|-------------|---------------------|
| QwenVL | Qwen2.5-VL multimodal large language model | `QWENVL_MODEL_PATH` |
| InternVL | InternVL3/2.5 vision-language model | `INTERNVL_MODEL_PATH` |
| Molmo | Allen AI's Molmo multimodal model | `MOLMO_MODEL_PATH` |
| VideoLLaMA | Video-focused large language model | `VIDEOLLAMA_MODEL_PATH` |
| MiniCPM | Efficient multimodal model | `MINICPM_MODEL_PATH` |
| Ovis | AIDC-AI's vision-language model | `OVIS_MODEL_PATH` |
| Phi | Microsoft's Phi multimodal model | `PHI_MODEL_PATH` |

## Supported Datasets

| Dataset | Description | Metric |
|---------|-------------|--------|
| VQAv2 | Visual Question Answering v2.0 | VQA Score |
| TextVQA | Text-based VQA | VQA Score |
| InfographicVQA | Infographic understanding | ANLS |
| ChartQA | Chart question answering | Relaxed Accuracy |
| ChartQAPro | Advanced chart QA | ChartQAPro Metric |
| DocVQA | Document question answering | ANLS |

## Installation

### Option 1: Full DAM-QA Installation (Recommended)

```bash
# Install DAM-QA with all dependencies
cd ..  # Go to project root
pip install -r requirements.txt
```

### Option 2: Evaluation Only

```bash
# Minimal installation for evaluation only
cd evaluation
pip install -r requirements.txt
```

## Model Configuration

### Option 1: Environment Variables (Recommended)

Set environment variables to point to your model paths:

```bash
export QWENVL_MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
export INTERNVL_MODEL_PATH="/path/to/InternVL2.5-8B" 
export MOLMO_MODEL_PATH="/path/to/Molmo-7B-D-0924"
# ... add other models as needed
```

### Option 2: Default HuggingFace Models

If no environment variables are set, models will be loaded from HuggingFace Hub using default model names.

## Dataset Configuration

This framework uses the unified dataset configuration from the main DAM-QA project. Dataset paths are automatically configured based on the `DAM_DATA_DIR` environment variable.

```bash
# Set your dataset directory (optional)
export DAM_DATA_DIR="/path/to/your/datasets"

# If not set, defaults to ../data/datasets/
```

The configuration is automatically loaded from the main project's `config.py`. For dataset setup instructions, see [../data/datasets.md](../data/datasets.md).

## Quick Start

### Single Model Evaluation

```bash
python run_vqa_eval.py --model qwenvl --dataset vqav2_val
```

### Batch Evaluation

```bash
# Edit evaluate_vqa.sh to configure models and datasets
./evaluate_vqa.sh
```

### Custom Configuration

```bash
python run_vqa_eval.py \
    --model internvl \
    --dataset infographicvqa_val \
    --batch-size 4 \
    --output-dir ./results/
```

## Configuration

### Dataset Configuration

Edit `config.py` to modify dataset paths, prompts, and evaluation metrics:

```python
ds_collections = {
    "your_dataset": {
        "data_path": "/path/to/dataset.jsonl",
        "image_folder": "/path/to/images/",
        "prompt": "Your custom prompt",
        "metric": "evaluation_metric",
        "max_new_tokens": 100,
    }
}
```

### Model Configuration

Each model has its own inference file in the `inference_models/` directory. To add a new model:

1. Create a new file: `inference_models/your_model.py`
2. Implement the `inference(questions, image_paths, config)` function
3. Follow the existing pattern for model initialization and inference

## Usage Examples

### Environment Variables

```bash
# Set GPU and batch size
export GPU=0
export BATCH_SIZE=8
./evaluate_vqa.sh
```

### Custom Model Evaluation

```python
from run_vqa_eval import load_inference_module

# Load your model's inference function
inference_fn = load_inference_module("your_model")

# Run inference
results = inference_fn(
    questions=["What is in the image?"],
    image_paths=["/path/to/image.jpg"],
    config={"max_new_tokens": 50}
)
```

## Output Format

Results are saved as CSV files with the following structure:

```csv
question_id,question,predict,gt,image_id
1,What color is the car?,red,red,123456
```

## Project Structure

```
evaluation/
├── run_vqa_eval.py          # Main evaluation script
├── config.py                # Dataset config adapter
├── evaluate_vqa.sh          # Batch evaluation script
├── score_vqa.py             # Scoring and metrics computation
├── auto_score.sh            # Automated scoring pipeline
├── setup_env.sh             # Environment setup helper
├── inference_models/        # Model inference modules
│   ├── qwenvl.py
│   ├── internvl.py
│   ├── molmo.py
│   ├── phi.py
│   └── ...
├── requirements.txt         # Evaluation dependencies
└── README.md               # This file
```

## Adding New Models

To add support for a new model:

1. Create `inference_models/new_model.py`
2. Implement the required interface:

```python
def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on your model.
    
    Args:
        questions: List of question strings
        image_paths: List of image file paths
        config: Configuration dictionary
    
    Returns:
        List of answer strings
    """
    # Your model inference code here
    return answers
```

## Adding New Datasets

1. Add dataset configuration to `config.py`:

```python
"new_dataset": {
    "data_path": "/path/to/data.jsonl",
    "image_folder": "/path/to/images/",
    "prompt": "Your prompt template",
    "metric": "evaluation_metric",
    "max_new_tokens": 100,
}
```

2. Ensure your data follows the expected JSONL format:

```json
{"question_id": 1, "question": "What is this?", "image": "image1.jpg", "answer": "cat"}
```

## Performance Tips

- Use appropriate batch sizes based on your GPU memory
- Consider using mixed precision (bfloat16) for faster inference
- Pre-load models to avoid repeated initialization overhead
- Use SSD storage for faster image loading

## Evaluation Metrics

- **VQA Score**: Standard VQA evaluation metric
- **ANLS**: Average Normalized Levenshtein Similarity
- **Relaxed Accuracy**: Allows minor numerical differences
- **ChartQAPro**: Specialized metric for chart understanding

## Integration with DAM-QA

This evaluation framework is designed to work seamlessly with the main DAM-QA codebase:

```bash
# From project root, run DAM-QA inference
python scripts/inference.py --method sliding --dataset infographicvqa_val

# Then evaluate the results with baseline models
cd evaluation
python run_vqa_eval.py --model qwenvl --dataset infographicvqa_val

# Score and compare results
python score_vqa.py --folder ../results/ --use_llm
```

## Citation

If you use this evaluation framework, please cite the DAM-QA paper:

```bibtex
@article{damqa2025,
  title={Describe Anything Model for Visual Question Answering on Text-rich Images},
  author={Vu, Yen-Linh and Duong, Dinh-Thang and Duong, Truong-Binh and Nguyen, Anh-Khoi and Nguyen, Le Thien Phuc and Xing, Jianhua and Li, Xingjian and Wang, Tianyang and Nguyen, Thanh-Huy and Bagci, Ulas and Xu, Min},
  journal={arXiv preprint},
  year={2025}
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Model Loading Errors**: Check model paths and ensure models are downloaded
3. **Image Loading Issues**: Verify image paths and formats
4. **Dependencies**: Ensure all required packages are installed with correct versions

### Getting Help

- Check the issues section for common problems
- Review model-specific documentation
- Ensure your environment meets the requirements

---

For more detailed information about specific models or datasets, please refer to their respective documentation and papers. 