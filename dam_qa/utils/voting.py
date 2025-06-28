"""
Voting utilities for combining predictions from multiple image regions
"""

from typing import Dict, Any
from collections import defaultdict


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


def calculate_window_weight(x0: int, y0: int, x1: int, y1: int, 
                          img_width: int, img_height: int) -> float:
    """
    Calculate the relative weight of a window based on its area.
    
    Args:
        x0, y0, x1, y1: Window coordinates
        img_width: Full image width
        img_height: Full image height
        
    Returns:
        Weight as fraction of total image area
    """
    window_area = (x1 - x0) * (y1 - y0)
    total_area = img_width * img_height
    return window_area / total_area 