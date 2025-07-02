"""
Common utility functions for DAM-QA experiments.
"""
import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
from collections import defaultdict


def resize_keep_aspect(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize PIL image so that its longest side = max_size, keeping aspect ratio.
    
    Args:
        img: PIL Image to resize
        max_size: Maximum size for the longest side
        
    Returns:
        Resized PIL Image
    """
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def create_full_image_mask(w: int, h: int) -> Image.Image:
    """
    Create a PIL grayscale mask where every pixel = 255 (foreground).
    DAM will consider the entire image as the region of interest.
    
    Args:
        w: Width of the image
        h: Height of the image
        
    Returns:
        PIL Image mask
    """
    mask_np = np.ones((h, w), dtype=np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def get_windows(W: int, H: int, window_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """
    Generate sliding window coordinates for image cropping.
    
    Args:
        W: Image width
        H: Image height
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        List of (x0, y0, x1, y1) coordinates
    """
    coords = []
    
    # Main sliding grid
    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            coords.append((x, y, x + window_size, y + window_size))
    
    # Right edge windows
    if coords and coords[-1][2] < W:
        for y in range(0, H - window_size + 1, stride):
            coords.append((W - window_size, y, W, y + window_size))
    
    # Bottom edge windows
    if coords and coords[-1][3] < H:
        for x in range(0, W - window_size + 1, stride):
            coords.append((x, H - window_size, x + window_size, H))
    
    # Remove duplicates
    return list(dict.fromkeys(coords))


def aggregate_votes(votes: defaultdict) -> str:
    """
    Aggregate votes from multiple predictions.
    
    Args:
        votes: Dictionary of prediction -> vote weight
        
    Returns:
        Best prediction based on weighted votes
    """
    if not votes:
        return ""
    
    return max(votes, key=votes.get)


def safe_load_image(img_path: str) -> Union[Image.Image, None]:
    """
    Safely load an image file.
    
    Args:
        img_path: Path to image file
        
    Returns:
        PIL Image or None if failed to load
    """
    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {img_path}")
        return None
    
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to load image {img_path}: {e}")
        return None


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory or file path
    """
    # If path ends with a file extension, get the directory
    if os.path.splitext(path)[1]:  # Has file extension
        path = os.path.dirname(path)
    # If path already exists as file, get its directory
    elif os.path.isfile(path):
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)


def format_question(question: Union[str, List]) -> str:
    """
    Format question from various input types.
    
    Args:
        question: Question as string or list
        
    Returns:
        Formatted question string
    """
    if isinstance(question, list):
        return " ".join(str(x) for x in question).strip()
    return str(question).strip() 