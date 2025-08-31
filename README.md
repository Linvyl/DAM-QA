# DAM-QA: Describe Anything Model for Visual Question Answering on Text-rich Images

[![Paper](https://img.shields.io/badge/arXiv-2507.12441-b31b1b.svg)](https://arxiv.org/abs/2507.12441)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/VLAI-AIVN/DAM-QA-annotations)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Linvyl/DAM-QA)

This repository contains the official implementation of **DAM-QA**, a framework that enhances Visual Question Answering (VQA) performance on text-rich images. Our approach extends the [Describe Anything Model (DAM)](https://github.com/NVlabs/describe-anything) by integrating a sliding-window mechanism with a weighted voting scheme to aggregate predictions from both global and local views.

![DAM-QA](assets/DAM-QA.png)

This method enables more effective grounding and reasoning over fine-grained textual information, leading to significant performance gains on challenging VQA benchmarks.

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Linvyl/DAM-QA.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Optional: Create a conda environment:**
   ```bash
   conda create -n dam-qa python=3.10
   conda activate dam-qa
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   - All required annotation `.jsonl` files are already included in the repository under the `data/` directory.
   - We also provide these unified annotation files in our [🤗 Hugging Face dataset repository](https://huggingface.co/datasets/VLAI-AIVN/DAM-QA-annotations) for convenience and reproducibility.
   - You only need to download the image files for each dataset. **Follow the instructions in [`data/dataset_guide.md`](data/dataset_guide.md)** to download and place the images in the correct subfolders.

   **⚠️ Important Note:** The annotation files are standardized conversions of existing public datasets (DocVQA, InfographicVQA, TextVQA, ChartQA, ChartQAPro, VQAv2) into a unified JSONL format following our experimental setup. These annotations preserve the original dataset content without modification. Please cite the original datasets appropriately when using them in your research.

## Repository Structure

```
DAM-QA/
├── src/                   # Core DAM-QA implementation
│   ├── config.py          # Dataset configs, prompts, parameters
│   ├── core.py            # Main inference classes
│   └── utils.py           # Utility functions
├── vlms/                  # VLM baseline implementations
│   ├── run_inference.py   # VLM inference runner
│   ├── config.py          # VLM dataset configurations
│   └── models/            # Individual VLM model implementations
│       ├── internvl.py    # InternVL3 model
│       ├── minicpm.py     # MiniCPM-o2.6 model
│       ├── molmo.py       # MolmoD model
│       ├── ovis.py        # OVIS2 model  
│       ├── phi.py         # Phi-4-Vision model
│       ├── qwenvl.py      # Qwen2.5-VL model
│       └── videollama.py  # VideoLLaMA3 model
├── evaluation/            # Evaluation framework
│   ├── metrics.py         # VQA scoring metrics
│   └── evaluator.py       # Main evaluation runner
├── run_experiment.py      # Main DAM-QA experiment runner
├── requirements.txt       # Python dependencies
├── data/                  # Datasets and annotation files (see below)
└── outputs/               # Results directory
    ├── full_image_default/
    ├── sliding_window_default/
    └── vlm_results/
```

## Datasets

Our implementation has been rigorously evaluated on the following benchmarks:

| Dataset            | Task                        | Metric           | Config Key |
| :----------------- | :-------------------------- | :--------------- | :----- |
| **DocVQA**         | Document Question Answering | ANLS             | `docvqa_val` |
| **InfographicVQA** | Infographic Understanding   | ANLS             | `infographicvqa_val` |
| **TextVQA**        | Scene-Text VQA              | VQA Score        | `textvqa_val` |
| **ChartQA**        | Chart Interpretation        | Relaxed Accuracy | `chartqa_test_human`, `chartqa_test_augmented` |
| **ChartQAPro**    | Advanced Chart QA           | Relaxed Accuracy | `chartqapro_test` |
| **VQAv2** (restval) | General-Purpose VQA         | VQA Score        | `vqav2_restval` |

## Data Preparation

### Dataset Structure

After downloading images as instructed in [`data/dataset_guide.md`](data/dataset_guide.md), your `data/` directory should look like this:

```
data/
├── docvqa/
│   ├── val.jsonl
│   └── images/
├── infographicvqa/
│   ├── infographicvqa_val.jsonl
│   └── images/
├── textvqa/
│   ├── textvqa_val_updated.jsonl
│   └── images/
├── chartqa/
│   ├── test_human.jsonl
│   ├── test_augmented.jsonl
│   └── images/
├── chartqapro/
│   ├── test.jsonl
│   └── images/
└── vqav2/
    ├── vqav2_restval.jsonl
    └── images/
```

- For detailed image download instructions, see [`data/dataset_guide.md`](data/dataset_guide.md).

## Running DAM-QA Experiments

### Basic Usage

Use `run_experiment.py` to run DAM-QA experiments:

**Full Image Baseline:**
```bash
python run_experiment.py --method full_image --dataset chartqapro_test --gpu 0
```

**Sliding Window (Our Method):**
```bash
python run_experiment.py --method sliding_window --dataset chartqapro_test --gpu 0
```

**Run on All Datasets:**
```bash
python run_experiment.py --method sliding_window --dataset all --gpu 0
```

### Ablation Studies

**Granularity Parameter Sweep:**
```bash
python run_experiment.py --method granularity_sweep --dataset chartqapro_test --gpu 0
```

**Prompt Design Ablation:**
```bash
python run_experiment.py --method prompt_ablation --dataset chartqapro_test --gpu 0
```

**Unanswerable Vote Weight Sweep:**
```bash
python run_experiment.py --method unanswerable_weight_sweep --dataset chartqapro_test --gpu 0
```

**Custom Parameters:**
```bash
python run_experiment.py \
    --method sliding_window \
    --dataset docvqa_val \
    --window_size 768 \
    --stride 384 \
    --unanswerable_weight 0.0 \
    --gpu 0
```

### Available Options

- `--method`: Choose from `full_image`, `sliding_window`, `granularity_sweep`, `prompt_ablation`, `unanswerable_weight_sweep`
- `--dataset`: Choose from `chartqapro_test`, `chartqa_test_human`, `docvqa_val`, `infographicvqa_val`, etc., or `all`
- `--window_size`: Sliding window size (default: 512)
- `--stride`: Sliding window stride (default: 256) 
- `--unanswerable_weight`: Weight for unanswerable votes (default: 0.0)
- `--use_visibility_rule`/`--no_visibility_rule`: Control visibility constraint
- `--use_unanswerable_rule`/`--no_unanswerable_rule`: Control unanswerable instruction

## Running VLM Baselines

### VLM Inference

Use `vlms/run_inference.py` to run VLM baseline models:

**InternVL:**
```bash
python vlms/run_inference.py --model internvl --dataset chartqapro_test
```

**Other supported models:** `minicpm`, `molmo`, `ovis`, `phi`, `qwenvl`, `videollama`

**Note:** If you encounter errors when running VLM models, install the required dependencies for each model:
- Follow installation instructions from the official HuggingFace or GitHub repositories of each VLM
- Each model may require specific versions of transformers, torch, or additional packages


## Evaluation

### Automatic Evaluation

Results are automatically saved to CSV files. Use the evaluation framework to compute metrics:

```bash
python evaluation/evaluator.py --folder ./outputs/sliding_window_default --use_llm
```

### Manual Score Calculation

```bash
python evaluation/metrics.py --file ./outputs/sliding_window_default/chartqapro_test/results.csv --use_llm
```

## Results

### Main Results

DAM-QA consistently outperforms the baseline DAM across multiple text-rich VQA benchmarks:

| Method            | DocVQA (ANLS) | InfographicVQA (ANLS) | TextVQA (VQA Score) | ChartQA (Relaxed Acc.) | ChartQAPro (Relaxed Acc.) | VQAv2 (VQA Score) |
| :---------------- | :-----------: | :-------------------: | :-----------------: | :--------------------: | :------------------------: | :---------------: |
| DAM (Baseline)    |     35.22     |         19.27         |        57.86        |          46.52          |           **18.90**            |       **79.25**       |
| **DAM-QA (Ours)** |   **42.34**   |       **20.25**       |      **59.67**      |       **47.72**        |           14.88             |      79.20     |

### Key Findings

- **Window Granularity**: Window size of 512 pixels with 50% overlap (stride=256) provides optimal performance
- **Prompt Design**: Both visibility constraint and unanswerable instruction are crucial
- **Vote Weighting**: Setting unanswerable weight to 0.0 significantly improves performance

## Configuration

### Main Configuration (`config.py`)

- **Model parameters**: Adjust `DEFAULT_INFERENCE_PARAMS` and `DEFAULT_IMAGE_PARAMS`
- **Experiment settings**: Modify `GRANULARITY_MODES` and `UNANSWERABLE_WEIGHTS`

### VLM Configuration

- **Dataset configurations**: Uses `DATASET_CONFIGS` from root `config.py`
- **Model-specific settings**: Configured in individual model files under `vlms/models/`


## Citation

```bibtex
@misc{vu2025modelvisualquestionanswering,
      title={Describe Anything Model for Visual Question Answering on Text-rich Images}, 
      author={Yen-Linh Vu and Dinh-Thang Duong and Truong-Binh Duong and Anh-Khoi Nguyen and Thanh-Huy Nguyen and Le Thien Phuc Nguyen and Jianhua Xing and Xingjian Li and Tianyang Wang and Ulas Bagci and Min Xu},
      year={2025},
      eprint={2507.12441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.12441}, 
}
```