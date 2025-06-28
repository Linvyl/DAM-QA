#!/bin/bash
#
# Batch evaluation script for DAM-QA
# Run inference on all supported datasets with the specified method.
#
# Usage:
#     bash scripts/run_all_datasets.sh [sliding|baseline] [gpu_id]
#
# Examples:
#     bash scripts/run_all_datasets.sh sliding 0
#     bash scripts/run_all_datasets.sh baseline 1
#

set -euo pipefail

# Configuration
METHOD=${1:-sliding}
GPU=${2:-0}
OUTPUT_DIR="./results/${METHOD}"

# Validate method
if [[ "$METHOD" != "sliding" && "$METHOD" != "baseline" ]]; then
    echo "Error: Method must be 'sliding' or 'baseline'"
    echo "Usage: $0 [sliding|baseline] [gpu_id]"
    exit 1
fi

# Dataset list
DATASETS=(
    "infographicvqa_val"
    "docvqa_val"
    "chartqa_test_human"
    "chartqa_test_augmented"
    "chartqapro_test"
    "textvqa_val"
    "vqav2_restval"
)

echo "========================================="
echo "DAM-QA Batch Evaluation"
echo "Method: $METHOD"
echo "GPU: $GPU"
echo "Output Directory: $OUTPUT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference on each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing dataset: $dataset"
    echo "-----------------------------------------"
    
    start_time=$(date +%s)
    
    python scripts/inference.py \
        --method "$METHOD" \
        --dataset "$dataset" \
        --output-dir "$OUTPUT_DIR" \
        --gpu "$GPU"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Completed $dataset in ${duration}s"
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "All datasets completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To evaluate results, run:"
echo "cd evaluation/eval_vqa"
echo "python score_vqa.py --folder $OUTPUT_DIR"
echo "=========================================" 