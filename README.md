# DAM-QA: Describe Anything Model for Visual Question Answering on Text-rich Images

This repository contains the official implementation of **DAM-QA**, a framework that enhances Visual Question Answering (VQA) performance on text-rich images. Our approach extends the [Describe Anything Model (DAM)](https://github.com/NVlabs/describe-anything) by integrating a sliding-window mechanism with a weighted voting scheme to aggregate predictions from both global and local views.

This method enables more effective grounding and reasoning over fine-grained textual information, leading to significant performance gains on challenging VQA benchmarks.


## Quick Start

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Linvyl/DAM-QA.git
    cd dam-qa
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Basic Usage

```python
from dam_qa import DAMSlidingWindow, DAMConfig

# Initialize the model with a specific configuration
config = DAMConfig(window_size=512, stride=256)
model = DAMSlidingWindow(config)

# Answer a question about an image
image_path = "path/to/your/image.jpg"
question = "What is the main title?"
answer = model.answer_question(image_path, question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Supported Datasets

Our implementation has been rigorously evaluated on the following benchmarks:

| Dataset          | Task                        | Metric            | Images |
| :--------------- | :-------------------------- | :---------------- | :----- |
| **DocVQA** | Document Question Answering | ANLS              | \~12K   |
| **InfographicVQA** | Infographic Understanding | ANLS              | \~5K    |
| **TextVQA** | Scene-Text VQA              | VQA Score         | \~28K   |
| **ChartQA** | Chart Interpretation        | Relaxed Accuracy  | \~20K   |
| **ChartQA-Pro** | Advanced Chart QA           | Relaxed Accuracy  | \~1.3K  |
| **VQAv2** | General-Purpose VQA         | VQA Score         | \~204K  |

## Data Preparation

### Download Datasets

Please download the datasets from their official sources:

1.  **DocVQA**: [Official Website](https://www.docvqa.org/)
2.  **InfographicVQA**: [Official Website](https://www.docvqa.org/datasets/infographicvqa)
3.  **TextVQA**: [Official Website](https://textvqa.org/)
4.  **ChartQA**: [Official Website](https://github.com/vis-nlp/ChartQA)
5.  **VQAv2**: [Official Website](https://visualqa.org/)

### Dataset Structure

For seamless integration, please organize your datasets according to the following directory structure:

```
/path/to/datasets/
├── DocVQA/
│   ├── val.jsonl
│   └── images/
├── InfographicVQA/
│   ├── infographicvqa_val.jsonl
│   └── images/
├── ChartQA/
│   ├── test_human.jsonl
│   ├── test_augmented.jsonl
│   └── images/
└── ...
```

### Configuration

DAM-QA supports environment-based configuration for dataset paths to ensure flexibility.

**1. Set Environment Variable (Optional)**

You can specify your main dataset directory using an environment variable. If not set, the framework defaults to `./data/datasets` within the project root.

```bash
export DAM_DATA_DIR="/path/to/your/datasets"
```

**2. Verify Setup**

The unified configuration is managed in `config.py`. To validate that your datasets are correctly configured, you can run:

```bash
# Check a specific dataset configuration
python -c "from config import validate_dataset_paths; print(validate_dataset_paths('infographicvqa_val'))"

# Validate the entire experimental setup
python experiments/utils.py
```

## Running Inference

### Command-Line Interface

Use `scripts/inference.py` to run DAM-QA on any supported dataset.

**DAM-QA (Our Sliding Window Method)**

```bash
python scripts/inference.py \
    --method sliding \
    --dataset docvqa_val \
    --output-dir ./results \
    --gpu 0
```

**Baseline (Full-Image Inference)**

```bash
python scripts/inference.py \
    --method baseline \
    --dataset docvqa_val \
    --output-dir ./results \
    --gpu 0
```

### Advanced Options

You can customize inference parameters such as window size, stride, and token limits:

```bash
python scripts/inference.py \
    --method sliding \
    --dataset chartqa_test_human \
    --window-size 768 \
    --stride 384 \
    --max-tokens 100 \
    --output-dir ./results
```

## Reproducing Results

### Main Results

To reproduce the main results reported in our paper, you can use the provided shell scripts, which execute inference across all supported datasets.

```bash
# Run evaluation using our sliding window approach
bash scripts/run_all_datasets.sh sliding

# Run evaluation using the baseline full-image approach
bash scripts/run_all_datasets.sh baseline
```

### Ablation Studies

Our repository includes scripts to reproduce the ablation studies.

1.  **Granularity Analysis** (`experiments/granularity/`):

    ```bash
    python experiments/granularity/dam_sliding_alldataset_granularity_sweep.py
    ```

2.  **Prompt Design** (`experiments/prompt_design/`):

    ```bash
    python experiments/prompt_design/dam_sliding_alldataset_no_rule.py
    python experiments/prompt_design/dam_sliding_alldataset_rule1_only.py
    python experiments/prompt_design/dam_sliding_alldataset_rule2_only.py
    ```

3.  **Vote Weighting** (`experiments/vote_weights/`):

    ```bash
    python experiments/vote_weights/dam_alldataset_find_best_unans_weight_sweep.py
    ```

## Results

### Main Results

DAM-QA consistently outperforms the baseline DAM across multiple text-rich VQA benchmarks.

| Method              | DocVQA (ANLS) | InfographicVQA (ANLS) | TextVQA (VQA Score) | ChartQA (Relaxed Acc.) | ChartQA-Pro (Relaxed Acc.) | VQAv2 (VQA Score) |
| :------------------ | :-----------: | :-------------------: | :-----------------: | :--------------------: | :------------------------: | :---------------: |
| DAM (Baseline)      |     35.22     |         19.27         |        65.22        |         35.62          |           18.90            |       79.25       |
| **DAM-QA (Ours)** |   **42.34** |       **35.60** |      **67.29** |       **40.06** |         **18.98** |     **79.20** |

### Ablation Study Highlights

  - **Window Granularity**: A window size of 512 pixels with a 50% overlap (stride of 256) provides the best trade-off between capturing fine-grained detail and maintaining global context.
  - **Prompt Design**: Enforcing both visual grounding (Rule 1) and abstention for insufficient evidence (Rule 2) yields the most balanced and robust performance.
  - **Vote Weighting**: Assigning a weight of zero to "unanswerable" predictions from local patches is critical. Including them in the vote aggregation dilutes correct answers and degrades performance.

## Repository Structure

```
DAM-QA/
├── dam_qa/                 # Main package for the DAM-QA framework
│   ├── models/             # Model implementations
│   │   ├── dam_sliding.py  # Sliding window approach
│   │   └── dam_baseline.py # Baseline full-image approach
│   ├── utils/              # Utility functions for image processing and voting
│   └── config.py           # Central configuration for datasets and model parameters
├── scripts/                # High-level scripts for inference and evaluation
│   └── inference.py        # Main inference script
├── experiments/            # Scripts to run ablation studies
│   ├── granularity/
│   ├── prompt_design/
│   └── vote_weights/
├── evaluation/             # Evaluation framework and scoring scripts
└── docs/                   # Documentation and related assets
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{damqa2025,
  title={Describe Anything Model for Visual Question Answering on Text-rich Images},
  author={Vu, Yen-Linh and Duong, Dinh-Thang and Duong, Truong-Binh and Nguyen, Anh-Khoi and Nguyen, Le Thien Phuc and Xing, Jianhua and Li, Xingjian and Wang, Tianyang and Nguyen, Thanh-Huy and Bagci, Ulas and Xu, Min},
  journal={arXiv preprint},
  year={2025}
}
```

**Keywords**: Visual Question Answering, Text-rich Images, Sliding Window, Document Understanding, Chart Analysis, Vision-Language Models.