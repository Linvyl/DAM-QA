#!/usr/bin/env bash
# VQA Model Evaluation Script
# This script runs evaluation for multiple models across multiple datasets

set -euo pipefail

# Configuration
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-1}

# Available models and datasets
AVAILABLE_MODELS=(qwenvl internvl molmo videollama minicpm ovis phi)
AVAILABLE_DATASETS=(vqav2_restval infographicvqa_val docvqa_val chartqa_test_human chartqa_test_augmented textvqa_val chartqapro_test)

# Current evaluation configuration
MODELS=(phi)
DATASETS=(infographicvqa_val)

# Uncomment lines below to run full evaluation:
# MODELS=(qwenvl internvl molmo videollama minicpm ovis phi)
# DATASETS=(vqav2_restval infographicvqa_val docvqa_val chartqa_test_human chartqa_test_augmented)

echo "Starting VQA evaluation..."
echo "GPU: $GPU"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Evaluating $model on $dataset..."
        
        CUDA_VISIBLE_DEVICES=$GPU python run_vqa_eval.py \
            --model "$model" \
            --dataset "$dataset" \
            --batch-size "$BATCH_SIZE"
        
        echo "Completed: $model on $dataset"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"
