#!/bin/bash
set -e

# Generate timestamped run directory
timestamp=$(date +"%Y-%m-%dT%H-%M-%S")
run_dir="sandbox/writable/run_$timestamp"

# Create the run directory on the host
mkdir -p "$run_dir"
echo "Created run directory: $run_dir"

# Set permissions if needed
chmod 777 "$run_dir"

# Export as environment variable for docker-compose to use
export EXPERIMENT_RUN_PATH="/app/sandbox/writable/run_$timestamp"
export HOST_RUN_DIR="$(pwd)/$run_dir"

echo "Starting container with run directory: $HOST_RUN_DIR"
echo "Container path: $EXPERIMENT_RUN_PATH"

# Run docker compose with the environment variables
docker compose --env-file .env -f infra/docker-compose.yml up --build -d ai-agent

echo "Container started. Run directory is at: $HOST_RUN_DIR"