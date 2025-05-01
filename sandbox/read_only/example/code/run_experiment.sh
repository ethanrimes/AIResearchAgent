#!/bin/bash
set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
MODEL_NAME="qwen3:1.7b"
MAX_PROBLEMS=5

# Print configuration
echo "Running LLM Math Reasoning Experiment"
echo "====================================="
echo "Results directory: $RESULTS_DIR"
echo "Model: $MODEL_NAME"
echo "Maximum problems to process: $MAX_PROBLEMS"
echo "====================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Ensure Python environment is set up
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check if Ollama server is running
if ! curl -s --max-time 5 "http://host.docker.internal:11434/api/tags" > /dev/null; then
    echo "Error: Cannot connect to Ollama server. Make sure it's running."
    exit 1
fi

# Run the experiment
echo "Starting experiment..."
python "${SCRIPT_DIR}/experiment_runner.py" \
    --output_dir "$RESULTS_DIR" \
    --model "$MODEL_NAME" \
    --max_problems "$MAX_PROBLEMS"

# Check if experiment completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Experiment failed."
    exit 1
fi

echo "Experiment completed successfully!"
echo "Results saved to: $RESULTS_DIR/results_summary.json"