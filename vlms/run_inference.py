import argparse
import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Callable
import importlib.util
from tqdm import tqdm
from collections import Counter
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from vlms.config import ds_collections


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def most_common_answer(answers: List[Dict[str, str]]) -> str:
    """Extract the most common answer from a list of answer dictionaries."""
    return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]


def load_inference_module(model_name: str) -> Callable:
    """Dynamically load inference function from model file."""
    model_file = Path(__file__).parent / "models" / f"{model_name}.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    spec = importlib.util.spec_from_file_location(model_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {model_name}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inference


def extract_image_id(item: Dict[str, Any], dataset: str) -> Any:
    """Extract image ID based on dataset format."""
    image_name = item["image"]
    
    if dataset == "vqav2_val":
        return int(image_name.split("_")[-1].split(".")[0])
    elif dataset == "infographicvqa_val":
        return int(image_name.split(".")[0])
    else:  # textvqa_val, chartqa, docvqa_val
        return image_name.split("/")[-1].split(".")[0]


def process_batch_data(data: List[Dict], dataset: str, config: Dict[str, Any]) -> tuple:
    """Process and extract batch data efficiently."""
    qids = [item["question_id"] for item in data]
    img_ids = [extract_image_id(item, dataset) for item in data]
    
    # Handle different question formats
    is_chartqapro = dataset == "chartqapro_test"
    questions = [item["question"][0] if is_chartqapro else item["question"] for item in data]
    gts = [item["answer"][0] if is_chartqapro else item["answer"] for item in data]
    
    img_paths = [os.path.join(config["image_folder"], item["image"]) for item in data]
    
    return qids, img_ids, questions, gts, img_paths


def save_results(data: List[Dict], model_name: str, dataset_name: str, output_dir: str = "./outputs/vlm_results/") -> None:
    """Save inference results to CSV file."""
    output_path = Path(output_dir) / f"{model_name}_{dataset_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode="w", encoding="utf-8", newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM Inference on VQA Datasets")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (file in models/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (key in ds_collections)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, 
                        default="./outputs/vlm_results/",
                        help="Output directory for results")
    
    args = parser.parse_args()
    set_seed()
    
    # Load configuration and inference function
    if args.dataset not in ds_collections:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration")
    
    config = ds_collections[args.dataset].copy()
    config["dataset_name"] = args.dataset
    inference_fn = load_inference_module(args.model)
    
    # Load and process data
    with open(config["data_path"], encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    qids, img_ids, questions, gts, img_paths = process_batch_data(data, args.dataset, config)
    
    # Datasets that use most common answer for ground truth
    vqa_datasets = {"vqav2_restval", "textvqa_val", "vqav2_val"}
    
    # Process batches
    for i in tqdm(range(0, len(questions), args.batch_size), desc="Processing batches"):
        batch_slice = slice(i, i + args.batch_size)
        batch_questions = questions[batch_slice]
        batch_images = img_paths[batch_slice]
        batch_gts = gts[batch_slice]
        
        # Run inference
        batch_answers = inference_fn(batch_questions, batch_images, config=config)
        
        # Display results
        print(f"Predictions: {batch_answers}")
        if args.dataset in vqa_datasets:
            gt_display = [most_common_answer(gt) for gt in batch_gts]
        else:
            gt_display = batch_gts
        print(f"Ground Truth: {gt_display}")
        
        # Update data with results
        for j, answer in enumerate(batch_answers):
            idx = i + j
            data[idx].update({
                "predict": answer,
                "gt": data[idx]["answer"],
                "image_id": img_ids[idx]
            })
            # Remove unnecessary fields
            data[idx].pop("image", None)
            data[idx].pop("answer", None)
    
    save_results(data, args.model, config["dataset_name"], args.output_dir)


if __name__ == "__main__":
    main()
