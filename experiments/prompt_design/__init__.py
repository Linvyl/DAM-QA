"""
Prompt design experiments for DAM-QA

🎯 **Primary Interface**: Use `run_experiments.py --experiment prompt --method [sliding|baseline]`

This module contains experiments evaluating the impact of different prompt
components on model performance.

Subdirectories:
- baseline/: Full-image inference experiments (empty - use unified interface)
- sliding_window/: Contains 1 clean example script for reference

Prompt variants available in unified interface:
- no_rules: Basic prompt without any rules
- rule1_only: Only visual grounding rule
- rule2_only: Only abstention rule  
- full: Complete prompt with all rules
"""

__all__ = [] 