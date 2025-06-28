# Dataset Organization

This folder contains the organized dataset files for DAM-QA evaluation.

## Expected Structure

```
data/datasets/
├── infographicvqa/
│   ├── infographicvqa_val.jsonl
│   └── images/
├── docvqa/
│   ├── val.jsonl
│   └── images/
├── chartqa/
│   ├── test_human.jsonl
│   ├── test_augmented.jsonl
│   └── images/
├── chartqapro/
│   ├── test.jsonl
│   └── images/
├── textvqa/
│   ├── textvqa_val_updated.jsonl
│   └── images/
└── vqav2/
    ├── vqav2_restval.jsonl
    └── images/
```

## Dataset Download

Please refer to `dataset.md` in the root directory for detailed instructions on downloading and preparing the datasets.

## Usage

After downloading and organizing your datasets according to the structure above, update the paths in:
- `dam_qa/config.py` - Main package configuration
- `evaluation/config.py` - Evaluation framework configuration

Set the paths to point to your local dataset locations. 