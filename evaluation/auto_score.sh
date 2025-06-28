#!/bin/bash

#=======================================================================
# Script to loop through all subfolders in a root directory and score them
#=======================================================================

ROOT_DIR="./results/csv_unscored"
GPU=0

PYTHON_SCRIPT="score_vqa.py"


echo "🚀 Start scoring..."

if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ Error: Root directory '$ROOT_DIR' not found."
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found. Make sure you are running this script from the same directory as the Python script."
    exit 1
fi


leaf_dirs=()
while IFS= read -r -d '' dir; do
    leaf_dirs+=("$dir")
done < <(find "$ROOT_DIR" -type d -links 2 -print0)

total=${#leaf_dirs[@]}

if [ "$total" -eq 0 ]; then
    echo "🟡 No subfolders found to score in '$ROOT_DIR'."
    exit 0
fi

echo "✅ Found $total subfolders to score."
echo "=================================================="

current=0
for dir in "${leaf_dirs[@]}"; do
    ((current++))

    printf "➡️  Progress: [%3d/%3d] | Scoring folder: %s\n" "$current" "$total" "${dir#$ROOT_DIR/}"
    CUDA_VISIBLE_DEVICES=$GPU python "$PYTHON_SCRIPT" --folder "$dir" --use_llm
    echo "--------------------------------------------------------------------------------------------------------"
done

echo "🎉 Done! Processed $total folders."