"""
DAM-QA: Describe Anything Model for Visual Question Answering on Text-rich Images

A sliding window approach for visual question answering that combines full-image 
and local patch information to improve performance on text-rich images.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models.dam_sliding import DAMSlidingWindow
from .models.dam_baseline import DAMBaseline
from .utils.common import (
    setup_device, load_dam_model, sliding_window_inference, 
    ExperimentLogger
)

__all__ = [
    "DAMSlidingWindow",
    "DAMBaseline",
    "setup_device",
    "load_dam_model",
    "sliding_window_inference",
    "ExperimentLogger",
] 