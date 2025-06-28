"""
Common utilities shared across DAM-QA experiments and inference.
This module consolidates frequently used functions to avoid code duplication.
"""

import os
import torch
from typing import List, Tuple, Union, Dict, Any
from PIL import Image
from transformers import AutoModel
from collections import defaultdict


def setup_device(device_id: Union[int, str] = 0) -> torch.device:
    """
    Setup CUDA device for inference.
    
    Args:
        device_id: CUDA device ID or device string
        
    Returns:
        torch.device object
    """
    if isinstance(device_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_id)
    
    return device


def load_dam_model(device: torch.device) -> Any:
    """
    Load DAM model with standard configuration.
    
    Args:
        device: Target device for model
        
    Returns:
        Initialized DAM model
    """
    dam_model = AutoModel.from_pretrained(
        "nvidia/DAM-3B-Self-Contained",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    
    return dam_model.init_dam(conv_mode="v1", prompt_mode="full+focal_crop")


def resize_keep_aspect(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize PIL image so that its longest side equals max_size, keeping aspect ratio.
    
    Args:
        img: PIL Image to resize
        max_size: Maximum size for the longest side
        
    Returns:
        Resized PIL Image
    """
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def get_sliding_windows(width: int, height: int, window_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """
    Generate sliding window coordinates for an image.
    
    Args:
        width: Image width
        height: Image height  
        window_size: Size of sliding window
        stride: Stride between windows
        
    Returns:
        List of (x0, y0, x1, y1) coordinates for each window
    """
    coords = []
    
    # Main sliding grid
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            coords.append((x, y, x + window_size, y + window_size))
    
    # Handle right edge if not covered
    if coords and coords[-1][2] < width:
        for y in range(0, height - window_size + 1, stride):
            coords.append((width - window_size, y, width, y + window_size))
    
    # Handle bottom edge if not covered
    if coords and coords[-1][3] < height:
        for x in range(0, width - window_size + 1, stride):
            coords.append((x, height - window_size, x + window_size, height))
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(coords))


def create_full_mask(width: int, height: int) -> Image.Image:
    """
    Create a full white mask for the entire image.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        PIL Image mask with all pixels set to 255 (white)
    """
    return Image.new("L", (width, height), 255)


def collect_votes(votes: Dict[str, float], answer: str, weight: float = 1.0, 
                 unanswerable_weight: float = 1.0) -> None:
    """
    Add a vote for an answer with the given weight.
    
    Args:
        votes: Dictionary to store votes
        answer: Answer string to vote for
        weight: Base weight for this vote
        unanswerable_weight: Additional weight multiplier for "unanswerable" answers
    """
    if answer and answer.strip():
        final_weight = weight
        if answer.lower() == "unanswerable":
            final_weight *= unanswerable_weight
        votes[answer] += final_weight


def get_final_prediction(votes: Dict[str, float], fallback: str = "") -> str:
    """
    Get the final prediction based on vote weights.
    
    Args:
        votes: Dictionary of answer -> weight
        fallback: Fallback answer if no votes
        
    Returns:
        The answer with the highest vote weight
    """
    if not votes:
        return fallback
    return max(votes, key=votes.get)


def dam_inference(dam_model: Any, image: Image.Image, mask: Image.Image, 
                 prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1e-7, top_p: float = 0.5, 
                 num_beams: int = 1) -> str:
    """
    Run DAM inference with standard parameters.
    
    Args:
        dam_model: Initialized DAM model
        image: PIL Image
        mask: PIL Image mask
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_beams: Number of beams for beam search
        
    Returns:
        Generated text response
    """
    tokens = dam_model.get_description(
        image, mask, prompt,
        streaming=False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens
    )
    
    return tokens.strip() if isinstance(tokens, str) else "".join(tokens).strip()


def sliding_window_inference(dam_model: Any, image: Image.Image, prompt: str,
                           window_size: int = 512, stride: int = 256,
                           max_new_tokens: int = 100, 
                           unanswerable_weight: float = 1.0) -> str:
    """
    Run sliding window inference on an image.
    
    Args:
        dam_model: Initialized DAM model
        image: PIL Image
        prompt: Text prompt
        window_size: Size of sliding windows
        stride: Stride between windows  
        max_new_tokens: Maximum tokens to generate
        unanswerable_weight: Weight multiplier for "unanswerable" responses
        
    Returns:
        Final prediction based on voting
    """
    W, H = image.size
    votes = defaultdict(float)
    
    # Full-image inference
    mask_full = create_full_mask(W, H)
    full_answer = dam_inference(dam_model, image, mask_full, prompt, max_new_tokens)
    if full_answer and full_answer != "unanswerable":
        collect_votes(votes, full_answer, weight=1.0, unanswerable_weight=unanswerable_weight)
    
    # Sliding window inference
    for (x0, y0, x1, y1) in get_sliding_windows(W, H, window_size, stride):
        crop = image.crop((x0, y0, x1, y1))
        mask_crop = create_full_mask(x1 - x0, y1 - y0)
        
        window_answer = dam_inference(dam_model, crop, mask_crop, prompt, max_new_tokens)
        if window_answer and window_answer != "unanswerable":
            # Weight by relative area
            weight = ((x1 - x0) * (y1 - y0)) / (W * H)
            collect_votes(votes, window_answer, weight=weight, unanswerable_weight=unanswerable_weight)
    
    return get_final_prediction(votes, fallback=full_answer)


class ExperimentLogger:
    """Simple logger for experiments."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = None
    
    def start(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        print(f"\n=== Starting {self.experiment_name} ===")
    
    def log(self, message: str, progress: int = None, total: int = None):
        """Log progress message."""
        if progress is not None and total is not None:
            print(f"[{progress}/{total}] {message}")
        else:
            print(f"[INFO] {message}")
    
    def finish(self, total_processed: int):
        """Log completion."""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            print(f"=== Completed {self.experiment_name} ===")
            print(f"Processed {total_processed} items in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        else:
            print(f"=== Completed {self.experiment_name} ===") 