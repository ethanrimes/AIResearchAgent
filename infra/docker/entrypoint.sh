#!/bin/bash
set -e

# Set default timeout (in seconds) for experiment execution
: "${EXPERIMENT_TIMEOUT:=300}"

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

# Verify write access to writable directories
if [ ! -w "/app/sandbox/writable/experiments" ]; then
    echo "Error: Cannot write to /app/sandbox/writable/experiments directory!"
    exit 1
else
    echo "✓ Write access to /app/sandbox/writable/experiments confirmed"
fi

if [ ! -w "/app/sandbox/writable/results" ]; then
    echo "Error: Cannot write to /app/sandbox/writable/results directory!"
    exit 1
else
    echo "✓ Write access to /app/sandbox/writable/results confirmed"
fi

if [ ! -w "/app/sandbox/writable/scratchpad" ]; then
    echo "Error: Cannot write to /app/sandbox/writable/scratchpad directory!"
    exit 1
else
    echo "✓ Write access to /app/sandbox/writable/scratchpad confirmed"
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