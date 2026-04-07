#!/bin/bash
set -e

# Configuration
MODEL_NAME="EgoMind-7B"
MODEL_PATH="models/${MODEL_NAME}"
OUTPUT_DIR="outputs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the model path is valid before starting
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' not found."
    exit 1
fi

# Refresh Ray cluster to ensure a clean state
echo "Restarting Ray cluster..."
ray stop && ray start --head

# List of benchmarks to evaluate
benchmarks=(
    "vsibench"
    "sitebench"
    "sparbench"
    "spbench"
)

# Main evaluation loop
for benchmark in "${benchmarks[@]}"; do
    echo "================================================"
    echo "Running evaluation for: ${benchmark}"
    echo "================================================"
    
    # Run evaluation and sync output to both console and log file
    python evaluation/run_eval.py \
        --model_path "$MODEL_PATH" \
        --output_path "${OUTPUT_DIR}/${MODEL_NAME}_${benchmark}.jsonl" \
        --benchmark "${benchmark}" \
    2>&1 | tee "${OUTPUT_DIR}/${MODEL_NAME}_${benchmark}.log"
done

echo "All evaluations completed successfully."