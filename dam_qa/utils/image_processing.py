"""
Image processing utilities for DAM-QA
"""

from typing import List, Tuple
from PIL import Image


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