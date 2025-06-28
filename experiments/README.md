# DAM-QA Experiments

This directory contains all experimental scripts for reproducing the ablation studies and analyses presented in the DAM-QA paper.

## 📁 Structure

```
experiments/
├── 🚀 run_experiments.py     # ⭐ Primary interface (USE THIS)
├── 🔧 utils.py              # Experiment utilities
├── 📖 README.md             # This documentation
├── granularity/             # Window size experiments
│   └── [1 clean example]    # ✅ Optimized, no duplicates
├── prompt_design/           # Prompt engineering studies
│   ├── baseline/            # Full-image experiments
│   └── sliding_window/      # Sliding window experiments
│       └── [1 clean example] # ✅ Optimized, no duplicates
└── vote_weights/            # Vote weighting experiments
    └── [1 clean example]    # ✅ Optimized, no duplicates
```

**🎯 All experiments now use the unified interface - no more duplicate code!**

## 🚀 Quick Start

### ⭐ Recommended: Unified Experiment Runner

Use the unified experiment runner for standardized, maintainable experiments:

```bash
# Check your setup first
python experiments/run_experiments.py --experiment setup

# Test different granularity settings
python experiments/run_experiments.py --experiment granularity

# Test prompt design variants
python experiments/run_experiments.py --experiment prompt --method sliding

# Test vote weighting strategies  
python experiments/run_experiments.py --experiment vote_weights

# Run on specific datasets only
python experiments/run_experiments.py --experiment granularity --datasets chartqa textvqa

# Use different inference methods
python experiments/run_experiments.py --experiment prompt --method baseline
```

### 📚 Example Scripts (Reference Only)

A few clean example scripts are provided for reference:
- `granularity/dam_sliding_alldataset_granularity_sweep.py`
- `prompt_design/sliding_window/dam_sliding_alldataset_no_rule.py` 
- `vote_weights/dam_alldataset_find_best_unans_weight_sweep.py`

⚠️ **These are for learning/reference only.** Use `run_experiments.py` for actual experiments.

## 🧪 Available Experiments

### 1. Granularity Analysis (`granularity/`)

Tests different window sizes and stride values to find optimal settings:
- **Coarse**: window_size=768, stride=384 (fewer large windows)
- **Fine**: window_size=256, stride=128 (many small windows)

### 2. Prompt Design Studies (`prompt_design/`)

Evaluates the impact of different prompt components:
- **no_rules**: Basic prompt without any rules
- **rule1_only**: Only visual grounding rule
- **rule2_only**: Only abstention rule
- **full**: Complete prompt with all rules

Available for both baseline (full-image) and sliding window methods.

### 3. Vote Weighting Analysis (`vote_weights/`)

Analyzes the effect of different weights for "unanswerable" responses:
- **equal**: 1.0x weight (baseline)
- **slight_boost**: 1.2x weight
- **medium_boost**: 1.5x weight  
- **strong_boost**: 2.0x weight

## 📊 Results

Results are automatically saved in `experiments/results/` with standardized naming:
- CSV files with predictions and ground truth
- Organized by experiment type and dataset
- Compatible with evaluation scripts

## 📋 Requirements

- GPU with 8GB+ memory recommended
- All datasets downloaded and configured (see `config.py`)
- DAM-QA package dependencies installed

## 🔧 Configuration

Update dataset paths in the main `config.py` file. Experiments automatically use the unified configuration system.

## 💡 Tips

- Use `--experiment setup` to validate your configuration before running experiments
- Monitor GPU memory usage during execution
- Results can be resumed if interrupted
- Use specific dataset selection to reduce runtime during development 