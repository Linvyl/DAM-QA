"""
Vote weighting experiments for DAM-QA

🎯 **Primary Interface**: Use `run_experiments.py --experiment vote_weights`

This module contains experiments analyzing the effect of different weights
for "unanswerable" responses in the voting mechanism.

Example script (reference only):
- dam_alldataset_find_best_unans_weight_sweep: Clean implementation example

Weight configurations available in unified interface:
- equal: 1.0x weight (baseline)
- slight_boost: 1.2x weight
- medium_boost: 1.5x weight  
- strong_boost: 2.0x weight
"""

__all__ = [] 