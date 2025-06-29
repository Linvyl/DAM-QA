# Dataset Image Download Guide for DAM-QA

This guide provides instructions for downloading **only the image files** for DAM-QA evaluation. The annotation JSONL files are already included in the repository.

## 🌟 Overview

DAM-QA supports evaluation on 6 VQA datasets that focus on text-rich image understanding:

- **DocVQA**: Document understanding and question answering
- **InfographicVQA**: Complex infographic understanding with text, charts, and visual elements
- **TextVQA**: Text-based visual question answering in natural scenes
- **ChartQA**: Chart and graph analysis (human + augmented variants)
- **ChartQA-Pro**: Advanced chart understanding with complex reasoning
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
# Navigate to DocVQA folder
cd data/docvqa

# Download and extract validation images
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
tar -zxvf val.tar.gz

# Move images to the expected location
mv val/* images/
rmdir val
rm val.tar.gz

cd ../..
```

### InfographicVQA Images

Download infographic images:

```bash
# Navigate to InfographicVQA folder  
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
# Navigate to TextVQA folder
cd data/textvqa

# Download training images (which include validation images)
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip

# Move train_images to images folder
mv train_images/* images/
rmdir train_images
rm train_val_images.zip

cd ../..
```

### ChartQA Images

Download chart images:

```bash
# Navigate to ChartQA folder
cd data/chartqa

# Download images from Google Drive
# Manual download required from: 
# https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view
# Extract the downloaded ChartQA_Dataset.zip to current folder
# Then move chart images:

# After manual download and extraction:
# mv ChartQA\ Dataset/test/* images/
# mv ChartQA\ Dataset/val/* images/
# mv ChartQA\ Dataset/train/* images/

cd ../..
```

**Note**: Manual download required from Google Drive link above.

### ChartQA-Pro Images

Download advanced chart images:

```bash
# Navigate to ChartQA-Pro folder
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
# Navigate to VQAv2 folder
cd data/vqav2

# Download COCO val2014 images (used by VQAv2)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# Move images to expected location
mv val2014/* images/
rmdir val2014
rm val2014.zip

cd ../..
```

## 📊 Dataset Statistics

| Dataset | Images | Questions | Domains | Metric |
|---------|--------|-----------|---------|--------|
| **DocVQA** | ~5K | 5,349 | Documents | ANLS |
| **InfographicVQA** | ~3K | 2,801 | Infographics | ANLS |
| **TextVQA** | ~5K | 5,000 | Natural scenes | VQA Score |
| **ChartQA** | ~1.25K×2 | 2,500 | Charts/Plots | Relaxed Accuracy |
| **ChartQA-Pro** | ~1.3K | 1,308 | Advanced charts | Relaxed Accuracy |
| **VQAv2** | ~40K | 214,354 | Natural images | VQA Score |

---

**Note**: The annotation JSONL files are already included in this repository under `data/{dataset}/`. You only need to download the image files following the instructions above.

For more information about DAM-QA, see the main [README](../README.md). 