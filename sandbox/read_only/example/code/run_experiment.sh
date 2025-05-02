#!/bin/bash
set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
RESULTS_DIR="run_2025-05-01T18-36-30/results"

# Print configuration
echo "Running LLM Math Reasoning Experiment"
echo "====================================="
echo "Results directory: $RESULTS_DIR"
echo "====================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Ensure Python environment is set up
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check if Ollama server is running
echo "Checking Ollama server connection..."
if ! curl -s --max-time 5 "http://host.docker.internal:11434/api/tags" > /dev/null; then
    echo "Error: Cannot connect to Ollama server. Make sure it's running."
    exit 1
fi
echo "Ollama server is available."

# Run the experiment
echo "Starting experiment..."
python "${SCRIPT_DIR}/experiment_runner.py"

# Check if experiment completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Experiment failed."
    exit 1
fi

# Verify that results file exists
if [ ! -f "${RESULTS_DIR}/results_summary.json" ]; then
    echo "Error: Results file not found at ${RESULTS_DIR}/results_summary.json"
    exit 1
fi

echo "Experiment completed successfully!"
echo "Results saved to: ${RESULTS_DIR}/results_summary.json"