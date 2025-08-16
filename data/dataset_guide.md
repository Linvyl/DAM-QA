# Dataset Image Download Guide for DAM-QA

This guide provides instructions for downloading **only the image files** for DAM-QA evaluation. The annotation JSONL files are already included in the repository under the `data/` directory and are also available on [🤗 Hugging Face](https://huggingface.co/datasets/VLAI-AIVN/DAM-QA-annotations).

## ⚠️ Dataset Attribution Notice

**Important:** The annotation files provided are standardized conversions of existing public datasets into a unified JSONL format. We do not claim ownership of the original dataset content. These files are:

- **DocVQA**: Converted from the original [DocVQA dataset](https://www.docvqa.org/)
- **InfographicVQA**: Converted from the original [InfographicVQA dataset](https://www.docvqa.org/datasets/infographicvqa)
- **TextVQA**: Converted from the original [TextVQA dataset](https://textvqa.org/)
- **ChartQA**: Converted from the original [ChartQA dataset](https://github.com/vis-nlp/ChartQA)
- **ChartQAPro**: Converted from the original [ChartQAPro dataset](https://huggingface.co/datasets/ahmed-masry/ChartQAPro)
- **VQAv2**: Converted from the original [VQAv2 dataset](https://visualqa.org/)

Please cite the original datasets appropriately when using them in your research. The conversion maintains data integrity while providing a consistent format for reproducible experiments.

## 🌟 Overview

DAM-QA supports evaluation on 6 VQA datasets that focus on text-rich image understanding:

- **DocVQA**: Document understanding and question answering
- **InfographicVQA**: Complex infographic understanding with text, charts, and visual elements
- **TextVQA**: Text-based visual question answering in natural scenes
- **ChartQA**: Chart and graph analysis (human + augmented variants)
- **ChartQAPro**: Advanced chart understanding with complex reasoning
- **VQAv2**: General visual question answering

## 📁 Expected Directory Structure

After downloading images, your directory structure should be:

```
data/
├── docvqa/
│   ├── val.jsonl               ✅ (included)
│   └── images/                 ⬇️ (download required)
├── infographicvqa/
│   ├── infographicvqa_val.jsonl ✅ (included)
│   └── images/                 ⬇️ (download required)
├── textvqa/
│   ├── textvqa_val_updated.jsonl ✅ (included)
│   └── images/                 ⬇️ (download required)
├── chartqa/
│   ├── test_human.jsonl        ✅ (included)
│   ├── test_augmented.jsonl    ✅ (included)
│   └── images/                 ⬇️ (download required)
├── chartqapro/
│   ├── test.jsonl              ✅ (included)
│   └── images/                 ⬇️ (download required)
└── vqav2/
    ├── vqav2_restval.jsonl     ✅ (included)
    └── images/                 ⬇️ (download required)
```

## 📥 Image Download Instructions

### DocVQA Images

Download document images from the official DocVQA dataset:

```bash
cd data/docvqa
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
tar -zxvf val.tar.gz
mv val/* images/
rmdir val
rm val.tar.gz
cd ../..
```

### InfographicVQA Images

Download infographic images:

```bash
cd data/infographicvqa
# Download images from official source
# Visit: https://rrc.cvc.uab.es/?ch=17&com=downloads
# Manual download required - get infographicsVQA validation images
# Extract to images/ folder
cd ../..
```
**Note**: Manual download required from [https://rrc.cvc.uab.es/?ch=17&com=downloads](https://rrc.cvc.uab.es/?ch=17&com=downloads)

### TextVQA Images

Download scene text images:

```bash
cd data/textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
mv train_images/* images/
rmdir train_images
rm train_val_images.zip
cd ../..
```

### ChartQA Images

Download chart images:

```bash
cd data/chartqa
# Download images from official repository
# https://github.com/vis-nlp/ChartQA
# Extract the downloaded ChartQA_Dataset.zip to current folder
# Then move chart images:
# mv ChartQA\ Dataset/test/* images/
# mv ChartQA\ Dataset/val/* images/
# mv ChartQA\ Dataset/train/* images/
cd ../..
```
**Note**: Manual download required from [https://github.com/vis-nlp/ChartQA](https://github.com/vis-nlp/ChartQA)

### ChartQAPro Images

Download advanced chart images:

```bash
cd data/chartqapro
# Download from official repository
# Visit: https://huggingface.co/datasets/ahmed-masry/ChartQAPro
# Follow their instructions to download images
# Extract to images/ folder
cd ../..
```
**Note**: Follow official instructions at [https://huggingface.co/datasets/ahmed-masry/ChartQAPro](https://huggingface.co/datasets/ahmed-masry/ChartQAPro)

### VQAv2 Images

Download COCO validation images for VQAv2:

```bash
cd data/vqav2
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
mv val2014/* images/
rmdir val2014
rm val2014.zip
cd ../..
```