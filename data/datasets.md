# Dataset Preparation Guide for DAM-QA

This guide provides detailed instructions for downloading and preparing the datasets required for DAM-QA evaluation.

## 📋 Supported Datasets

DAM-QA has been evaluated on the following Visual Question Answering benchmarks:

| Dataset | Description | Task | Metric | Size |
|---------|-------------|------|--------|------|
| **DocVQA** | Document Understanding | Document QA | ANLS | ~12K questions |
| **InfographicVQA** | Infographic Analysis | Chart/Visual QA | ANLS | ~5K questions |
| **TextVQA** | Scene Text Reading | Text-based VQA | VQA Score | ~28K questions |
| **ChartQA** | Chart Interpretation | Chart QA | Relaxed Accuracy | ~20K questions |
| **ChartQA-Pro** | Advanced Chart Analysis | Advanced Chart QA | Relaxed Accuracy | ~1.3K questions |
| **VQAv2** | General Visual QA | General VQA | VQA Score | ~204K questions |

## 🔧 Quick Setup

### 1. Set Environment Variable (Optional)

```bash
export DAM_DATA_DIR="/path/to/your/datasets"
```

If not set, datasets will be expected in `./data/datasets/` relative to the project root.

### 2. Create Directory Structure

```bash
mkdir -p data/datasets/{docvqa,infographicvqa,textvqa,chartqa,chartqapro,vqav2}
```

## 📥 Dataset Download Instructions

### DocVQA

**Official Website**: [https://www.docvqa.org/](https://www.docvqa.org/)

1. Visit the official website and register for access
2. Download the validation set:
   - `val.jsonl` - Question-answer pairs
   - `images/` - Document images

3. Expected structure:
   ```
   data/datasets/docvqa/
   ├── val.jsonl
   └── images/
       ├── document1.png
       ├── document2.png
       └── ...
   ```

### InfographicVQA

**Official Website**: [https://www.docvqa.org/datasets/infographicvqa](https://www.docvqa.org/datasets/infographicvqa)

1. Register and download the validation set:
   - `infographicvqa_val.jsonl` - Question-answer pairs
   - `images/` - Infographic images

2. Expected structure:
   ```
   data/datasets/infographicvqa/
   ├── infographicvqa_val.jsonl
   └── images/
       ├── infographic1.png
       ├── infographic2.png
       └── ...
   ```

### TextVQA

**Official Website**: [https://textvqa.org/](https://textvqa.org/)

1. Download the validation set:
   - `textvqa_val_updated.jsonl` - Question-answer pairs
   - `images/` - Scene text images (subset of OpenImages)

2. Expected structure:
   ```
   data/datasets/textvqa/
   ├── textvqa_val_updated.jsonl
   └── images/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

### ChartQA

**Official Website**: [https://github.com/vis-nlp/ChartQA](https://github.com/vis-nlp/ChartQA)

1. Clone the repository or download the test sets:
   - `test_human.jsonl` - Human-generated questions
   - `test_augmented.jsonl` - Machine-generated questions
   - `images/` - Chart images

2. Expected structure:
   ```
   data/datasets/chartqa/
   ├── test_human.jsonl
   ├── test_augmented.jsonl
   └── images/
       ├── chart1.png
       ├── chart2.png
       └── ...
   ```

### ChartQA-Pro

**Official Website**: [https://chartqapro.github.io/](https://chartqapro.github.io/)

1. Follow the official instructions to download:
   - `test.jsonl` - Test questions with advanced reasoning
   - `images/` - Chart images

2. Expected structure:
   ```
   data/datasets/chartqapro/
   ├── test.jsonl
   └── images/
       ├── chart1.png
       ├── chart2.png
       └── ...
   ```

### VQAv2

**Official Website**: [https://visualqa.org/](https://visualqa.org/)

1. Download the validation set:
   - `vqav2_restval.jsonl` - Restval question-answer pairs
   - `images/` - COCO validation images

2. Expected structure:
   ```
   data/datasets/vqav2/
   ├── vqav2_restval.jsonl
   └── images/
       ├── COCO_val2014_000000000001.jpg
       ├── COCO_val2014_000000000002.jpg
       └── ...
   ```

## 📝 Data Format

All datasets should follow the JSONL format with the following fields:

```json
{
  "question_id": "unique_identifier",
  "question": "What is shown in the image?",
  "image": "image_filename.jpg",
  "answer": ["ground_truth_answer"]
}
```

**Notes:**
- `answer` can be a string or list depending on the dataset
- Some datasets may have additional fields like `question_type`
- Image paths are relative to the dataset's `images/` folder

## ✅ Validation

After downloading, validate your setup:

```python
from config import validate_dataset_paths

# Check a specific dataset
result = validate_dataset_paths('docvqa_val')
print(result)

# Check all datasets
from config import list_available_datasets
for dataset in list_available_datasets():
    result = validate_dataset_paths(dataset)
    if result['data_file_exists'] and result['image_folder_exists']:
        print(f"✅ {dataset}: Ready")
    else:
        print(f"❌ {dataset}: Missing files")
```

## 🚀 Quick Test

Test your setup with a small inference run:

```bash
# Test with a single dataset
python scripts/inference.py \
    --method sliding \
    --dataset docvqa_val \
    --output-dir ./test_results \
    --gpu 0

# Check the output
ls ./test_results/
```

## 📊 Dataset Statistics

| Dataset | Train | Val | Test | Languages | Domains |
|---------|--------|-----|------|-----------|---------|
| DocVQA | 39,463 | 5,349 | 5,188 | EN | Documents |
| InfographicVQA | 23,946 | 2,801 | 3,288 | EN | Infographics |
| TextVQA | 34,602 | 5,000 | 5,734 | EN | Natural scenes |
| ChartQA | 18,271 | 1,250 | 1,250 | EN | Charts/Plots |
| ChartQA-Pro | - | - | 1,308 | EN | Advanced charts |
| VQAv2 | 443,757 | 214,354 | 447,793 | EN | Natural images |

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Verify the dataset directory structure matches the expected format
   - Check file permissions and paths
   - Ensure JSONL files are properly formatted

2. **Image Loading Issues**
   - Verify image file extensions (.jpg, .png, etc.)
   - Check that image paths in JSONL match actual filenames
   - Ensure sufficient disk space

3. **Permission Errors**
   - Some datasets require registration and agreement to terms
   - Make sure you have proper access credentials

### Getting Help

1. Check the dataset's official documentation
2. Verify your directory structure matches the examples above
3. Test with a small subset of data first
4. Open an issue if you encounter dataset-specific problems

## 📚 References

If you use these datasets, please cite the original papers:

- **DocVQA**: Mathew et al., "DocVQA: A Dataset for VQA on Document Images", WACV 2021
- **InfographicVQA**: Mathew et al., "InfographicVQA", WACV 2022  
- **TextVQA**: Singh et al., "Towards VQA Models That Can Read", CVPR 2019
- **ChartQA**: Masry et al., "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning", ACL 2022
- **VQAv2**: Goyal et al., "Making the V in VQA Matter", CVPR 2017

---

For more information about DAM-QA, see the main [README](../README.md). 