#!/bin/bash
set -e

# Extract timeout from default_config.yaml
CONFIG_TIMEOUT=$(python -c "
import yaml
with open('core/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('max_run_duration', 600))
")

# Export the timeout value from config
export EXPERIMENT_TIMEOUT="${CONFIG_TIMEOUT}"
echo "Using timeout from config: ${EXPERIMENT_TIMEOUT}s"


# Generate timestamped run directory
timestamp=$(date +"%Y-%m-%dT%H-%M-%S")
run_dir="sandbox/writable/run_$timestamp"

# Create the run directory on the host
mkdir -p "$run_dir"
echo "Created run directory: $run_dir"

# Copy pyright configuration to experiment directory
cp core/config/package_config/pyrightconfig.json "$run_dir/pyrightconfig.json"

# Set permissions if needed
chmod 777 "$run_dir"

# Set these variables for use throughout the script
EXPERIMENT_RUN_PATH="/app/sandbox/writable/run_$timestamp"
HOST_RUN_DIR="$(pwd)/$run_dir"

# Export them for docker-compose
export EXPERIMENT_RUN_PATH
export HOST_RUN_DIR

echo "Starting container with run directory: $HOST_RUN_DIR"
echo "Container path: $EXPERIMENT_RUN_PATH"

# Check if build argument is passed
BUILD_FLAG=""
if [[ "$1" == "build" ]]; then
  echo "Build flag detected. Will rebuild the Docker image."
  BUILD_FLAG="--build"
  
  # Force remove the existing image if it exists
  echo "Removing existing ai-research-agent images..."
  docker compose --env-file .env -f infra/docker-compose.yml down
  
  # Remove the image to ensure a clean build
  docker rmi $(docker images -q ai-research-agent:latest) 2>/dev/null || true
  
  # Also remove any dangling images (optional cleanup)
  docker image prune -f
  
  echo "Building new image..."
  # Force build with no-cache to ensure fresh build
  docker compose --env-file .env -f infra/docker-compose.yml build --no-cache ai-agent
  
  # Add the requested command to remove exec.cachemount
  docker builder prune --filter type=exec.cachemount --force
fi

# Setup trap to clean up when script exits
function cleanup {
  echo "Cleaning up containers..."
  # Make sure to re-export the variables within the function
  export EXPERIMENT_RUN_PATH="$EXPERIMENT_RUN_PATH"
  export HOST_RUN_DIR="$HOST_RUN_DIR"
  docker compose --env-file .env -f infra/docker-compose.yml down
  echo "Containers removed."
}

# Register the cleanup function to run on exit
trap cleanup EXIT

# Run docker compose with the environment variables
echo "Starting container..."
docker compose --env-file .env -f infra/docker-compose.yml up $BUILD_FLAG ai-agent

echo "Container started. Run directory is at: $HOST_RUN_DIR"
# The cleanup function will run automatically when the script exits

# After building the new image
echo "Cleaning up dangling images..."
docker image prune -f