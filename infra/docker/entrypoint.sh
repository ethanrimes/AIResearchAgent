#!/bin/bash
set -e

# Set default timeout
: "${EXPERIMENT_TIMEOUT:=300}"

# Use the run path provided from the environment
: "${EXPERIMENT_RUN_PATH:=/app/sandbox/writable/run_default}"

echo "Experiment run directory: $EXPERIMENT_RUN_PATH"

# Write to a file so Python can read it too (if not passed through env)
echo "$EXPERIMENT_RUN_PATH" > /app/active_run_path.txt

# Print startup information
echo "Starting AI Research Agent with experiment timeout: ${EXPERIMENT_TIMEOUT}s"
echo "Using OpenAI API for agent"
echo "Using Ollama for local inference via host.docker.internal"

# Check permissions
echo "Checking filesystem permissions..."

# Verify read access to read_only directory
if [ ! -r "/app/sandbox/read_only" ]; then
    echo "Error: Cannot read from /app/sandbox/read_only directory!"
    exit 1
else
    echo "✓ Read access to /app/sandbox/read_only confirmed"
fi

# Verify read access to data directory
if [ ! -r "/app/data" ]; then
    echo "Error: Cannot read from /app/data directory!"
    exit 1
else
    echo "✓ Read access to /app/data confirmed"
fi

# The run directory should already exist, but create subdirectories
mkdir -p "$EXPERIMENT_RUN_PATH"/{planning,experiment,results,scratchpad}

# Verify write access to the specific run directory
if [ ! -w "$EXPERIMENT_RUN_PATH" ]; then
    echo "Error: Cannot write to $EXPERIMENT_RUN_PATH directory!"
    exit 1
else
    echo "✓ Write access to $EXPERIMENT_RUN_PATH confirmed"
fi

# Verify we CANNOT write to the parent writable directory (should be read-only)
if touch /app/sandbox/writable/test_write.txt 2>/dev/null; then
    echo "Error: Unexpected write access to /app/sandbox/writable!"
    rm -f /app/sandbox/writable/test_write.txt
    exit 1
else
    echo "✓ Confirmed /app/sandbox/writable is read-only as expected"
fi

# Test network connectivity to OpenAI API
echo "Testing connectivity to OpenAI API..."
if curl -s --max-time 5 https://api.openai.com/v1/models > /dev/null; then
    echo "✓ Network connectivity to OpenAI API confirmed"
else
    echo "Warning: Cannot connect to OpenAI API, check your network configuration"
fi

# Test network connectivity to Ollama
echo "Testing connectivity to Ollama..."
OLLAMA_HOST=$(echo $OLLAMA_BASE_URL | sed 's|^[^/]*//||' | sed 's|/.*$||')
if curl -s --max-time 5 $OLLAMA_BASE_URL/api/tags > /dev/null; then
    echo "✓ Network connectivity to Ollama confirmed"
else
    echo "Warning: Cannot connect to Ollama at $OLLAMA_BASE_URL"
    echo "Make sure Ollama is running on your host machine"
fi

# Run the main experiment script with timeout
if [ -f "/app/core/scripts/run_experiment.py" ]; then
    echo "Executing experiment with ${EXPERIMENT_TIMEOUT}s timeout..."
    timeout ${EXPERIMENT_TIMEOUT} python /app/core/scripts/run_experiment.py
else
    echo "Error: run_experiment.py not found!"
    exit 1
fi