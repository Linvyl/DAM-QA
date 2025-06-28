#!/bin/bash

# Environment Variable Setup Script for DAM-QA Evaluation Framework
# This script helps you configure model paths for the evaluation framework

echo "🚀 Setting up environment variables for DAM-QA evaluation..."
echo ""

# Default model paths (modify these to match your setup)
DEFAULT_QWENVL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_INTERNVL_PATH="OpenGVLab/InternVL2.5-8B"
DEFAULT_MOLMO_PATH="allenai/Molmo-7B-D-0924"
DEFAULT_VIDEOLLAMA_PATH="DAMO-NLP-SG/VideoLLaMA2-7B-Base"
DEFAULT_MINICPM_PATH="openbmb/MiniCPM-o-2_6"
DEFAULT_OVIS_PATH="AIDC-AI/Ovis1.6-Gemma2-9B"
DEFAULT_PHI_PATH="microsoft/Phi-3.5-vision-instruct"

# Function to set environment variable with user input
set_env_var() {
    local var_name=$1
    local default_value=$2
    local current_value=${!var_name}
    
    echo "Current $var_name: ${current_value:-'(not set)'}"
    echo "Default: $default_value"
    read -p "Enter path for $var_name (press Enter for default): " user_input
    
    if [ -z "$user_input" ]; then
        export $var_name="$default_value"
        echo "Set $var_name to default: $default_value"
    else
        export $var_name="$user_input"
        echo "Set $var_name to: $user_input"
    fi
    echo ""
}

echo "Configure model paths (you can use local paths or HuggingFace model names):"
echo ""

# Set up each model path
set_env_var "QWENVL_MODEL_PATH" "$DEFAULT_QWENVL_PATH"
set_env_var "INTERNVL_MODEL_PATH" "$DEFAULT_INTERNVL_PATH"
set_env_var "MOLMO_MODEL_PATH" "$DEFAULT_MOLMO_PATH"
set_env_var "VIDEOLLAMA_MODEL_PATH" "$DEFAULT_VIDEOLLAMA_PATH"
set_env_var "MINICPM_MODEL_PATH" "$DEFAULT_MINICPM_PATH"
set_env_var "OVIS_MODEL_PATH" "$DEFAULT_OVIS_PATH"
set_env_var "PHI_MODEL_PATH" "$DEFAULT_PHI_PATH"

# Generate export commands for future use
echo "✅ Environment variables set for current session!"
echo ""
echo "To make these permanent, add the following to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "# DAM-QA Model Paths"
echo "export QWENVL_MODEL_PATH=\"$QWENVL_MODEL_PATH\""
echo "export INTERNVL_MODEL_PATH=\"$INTERNVL_MODEL_PATH\""
echo "export MOLMO_MODEL_PATH=\"$MOLMO_MODEL_PATH\""
echo "export VIDEOLLAMA_MODEL_PATH=\"$VIDEOLLAMA_MODEL_PATH\""
echo "export MINICPM_MODEL_PATH=\"$MINICPM_MODEL_PATH\""
echo "export OVIS_MODEL_PATH=\"$OVIS_MODEL_PATH\""
echo "export PHI_MODEL_PATH=\"$PHI_MODEL_PATH\""
echo ""
echo "📋 You can now run evaluations with:"
echo "python run_vqa_eval.py --model qwenvl --dataset infographicvqa_val" 