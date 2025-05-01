#!/bin/bash
set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="${BASE_DIR}/sandbox/writable/run_$(date +%Y-%m-%dT%H-%M-%S)/results"
DATASET_PATH="${BASE_DIR}/data/rstar/aime2024_test.json"
MODEL_NAME="qwen3:1.7b"
MAX_PROBLEMS=5

# Print configuration
echo "Running LLM Reasoning Enhancement Experiment"
echo "============================================="
echo "Base directory: $BASE_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_NAME"
echo "Maximum problems to process: $MAX_PROBLEMS"
echo "============================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Ensure Python environment is set up
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Check if Ollama server is running
if ! curl -s --max-time 5 "http://host.docker.internal:11434/api/tags" > /dev/null; then
    echo "Error: Cannot connect to Ollama server. Make sure it's running."
    exit 1
fi

# Check if model is available
if ! curl -s "http://host.docker.internal:11434/api/tags" | grep -q "$MODEL_NAME"; then
    echo "Warning: Model $MODEL_NAME not found in Ollama. The experiment may fail."
    echo "Available models:"
    curl -s "http://host.docker.internal:11434/api/tags" | grep "name" | cut -d'"' -f4
fi

# Run the experiment
echo "Starting experiment..."
python "${SCRIPT_DIR}/experiment_runner.py" \
    --dataset "$DATASET_PATH" \
    --output-dir "$RESULTS_DIR" \
    --model "$MODEL_NAME" \
    --max-problems "$MAX_PROBLEMS"

# Check if experiment completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Experiment failed."
    exit 1
fi

# Generate visualizations
echo "Generating visualizations..."
python -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from utils import create_results_visualizations, load_json
results = load_json('${RESULTS_DIR}/results_summary.json')
create_results_visualizations(results, '${RESULTS_DIR}/visualizations')
"

# Generate markdown summary
echo "Generating experiment summary..."
python -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from utils import format_experiment_summary, load_json, save_json
results = load_json('${RESULTS_DIR}/results_summary.json')
summary = format_experiment_summary(results)
with open('${RESULTS_DIR}/experiment_summary.md', 'w') as f:
    f.write(summary)
"

echo "Experiment completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo "To view the summary, open: $RESULTS_DIR/experiment_summary.md"