# Dataset Download and Preparation Guide

This guide provides comprehensive instructions for downloading and organizing the datasets used in DAM-QA evaluation.

## 🌟 Overview

DAM-QA supports evaluation on 6 VQA datasets that focus on text-rich image understanding:

- **InfographicVQA**: Complex infographic understanding
- **DocVQA**: Document question answering  
- **ChartQA**: Chart and graph analysis
- **ChartQA-Pro**: Advanced chart understanding
- **TextVQA**: Text-based visual question answering
- **VQAv2**: General visual question answering

## 📁 Target Directory Structure

After following this guide, your datasets should be organized as follows:

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

## 🗂️ Dataset Preparation

### InfographicVQA

InfographicVQA focuses on understanding complex infographics with text, charts, and visual elements.

```bash
# Create directory
mkdir -p data/datasets/infographicvqa/images

# Download from official source
# Visit: https://rrc.cvc.uab.es/?ch=17&com=downloads
# Download: infographicsVQA_val_v1.0_withQT.json and images

# Convert to JSONL format (you may need to implement conversion script)
# Expected format: {"question_id": int, "question": str, "image": str, "answer": str}
```

**Official Links:**
- Paper: https://arxiv.org/abs/2104.12756
- Dataset: https://rrc.cvc.uab.es/?ch=17&com=downloads

### DocVQA

DocVQA focuses on document understanding and question answering.

```bash
# Create directory
mkdir -p data/datasets/docvqa/images

# Download from official source
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
tar -zxvf val.tar.gz -C data/datasets/docvqa/

# Convert to JSONL format
# Expected format: {"question_id": int, "question": str, "image": str, "answer": str}
```

**Official Links:**
- Paper: https://arxiv.org/abs/2007.00398
- Dataset: https://rrc.cvc.uab.es/?ch=17&com=downloads

### ChartQA

ChartQA evaluates chart understanding capabilities with two variants.

```bash
# Create directory
mkdir -p data/datasets/chartqa/images

# Download images from Google Drive
# Link: https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view
# Extract to data/datasets/chartqa/images/

# Download question files
wget https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/test_human.json
wget https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/test_augmented.json

# Convert to JSONL format
# Expected format: {"question_id": int, "question": str, "image": str, "answer": str}
```

**Official Links:**
- Paper: https://arxiv.org/abs/2203.10244
- Dataset: https://github.com/vis-nlp/ChartQA

### ChartQA-Pro

Advanced chart understanding benchmark with complex reasoning.

```bash
# Create directory
mkdir -p data/datasets/chartqapro/images

# Download from official repository
# Visit: https://github.com/your-org/chartqa-pro
# Download: test.jsonl and images folder

# No conversion needed if already in JSONL format
```


### TextVQA

TextVQA focuses on reading and reasoning about text in images.

```bash
# Create directory
mkdir -p data/datasets/textvqa/images

# Download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d data/datasets/textvqa/
mv data/datasets/textvqa/train_images data/datasets/textvqa/images

# Download annotations
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json

# Convert to JSONL format
# Expected format: {"question_id": int, "question": str, "image": str, "answer": [list of answers]}
```

**Official Links:**
- Paper: https://arxiv.org/abs/1904.08920
- Dataset: https://textvqa.org/

### VQAv2

General visual question answering benchmark.

```bash
# Create directory
mkdir -p data/datasets/vqav2/images

# Download COCO images (VQAv2 uses COCO val2014)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/datasets/vqav2/
mv data/datasets/vqav2/val2014/* data/datasets/vqav2/images/

# Download VQAv2 annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip

# Convert to JSONL format  
# Expected format: {"question_id": int, "question": str, "image": str, "answer": [list of answer dicts]}
```

**Official Links:**
- Paper: https://arxiv.org/abs/1612.00837
- Dataset: https://visualqa.org/
