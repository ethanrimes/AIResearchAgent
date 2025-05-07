#!/bin/bash
set -euo pipefail

# -------------------------------------------------
# 0.  Read environment variables and config file
# -------------------------------------------------
# Extract timeout from YAML config file with Python
CONFIG_TIMEOUT=$(python -c "
import yaml
with open('/app/core/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('max_run_duration', 600))
")

# Set default if parsing failed or value is not present
: "${EXPERIMENT_TIMEOUT:=${CONFIG_TIMEOUT}}"
: "${EXPERIMENT_RUN_PATH:=/app/sandbox/writable/run_default}"

echo "Experiment run directory: $EXPERIMENT_RUN_PATH"
echo "$EXPERIMENT_RUN_PATH" > /app/active_run_path/path.txt

echo "Starting AI Research Agent with experiment timeout: ${EXPERIMENT_TIMEOUT}s"
echo "Using OpenAI API for agent"
echo "Using Ollama for local inference via host.docker.internal"

# -------------------------------------------------
# 1.  Filesystem-permission checks
# -------------------------------------------------
echo "Checking filesystem permissions..."

readonly_ro="/app/sandbox/read_only"
readonly_data="/app/data"
readonly_parent="/app/sandbox/writable"

if [[ ! -r "$readonly_ro" ]];   then echo "❌ Cannot read $readonly_ro";   exit 1; else echo "✓ Read access to $readonly_ro confirmed"; fi
if [[ ! -r "$readonly_data" ]]; then echo "❌ Cannot read $readonly_data"; exit 1; else echo "✓ Read access to $readonly_data confirmed"; fi

# Create run sub-dirs (the bind mount itself must already exist)
mkdir -p "${EXPERIMENT_RUN_PATH}"/{hypothesis,planning,experiment,results,findings,scratchpad,logs}
touch "${EXPERIMENT_RUN_PATH}"/experiment/run_experiment.sh
if [[ ! -w "$EXPERIMENT_RUN_PATH" ]]; then
  echo "❌ Cannot write to $EXPERIMENT_RUN_PATH"; exit 1
else
  echo "✓ Write access to $EXPERIMENT_RUN_PATH confirmed"
fi

# We must NOT be able to write to the parent writable dir
if touch "${readonly_parent}/should_not_write.txt" 2>/dev/null; then
  echo "❌ Unexpected write access to $readonly_parent"; rm -f "${readonly_parent}/should_not_write.txt"; exit 1
else
  echo "✓ Confirmed $readonly_parent is read-only as expected"
fi

# -------------------------------------------------
# 2.  Network-connectivity smoke tests
# -------------------------------------------------
echo "Testing connectivity to OpenAI API..."
if curl -s --max-time 5 https://api.openai.com/v1/models >/dev/null; then
  echo "✓ Network connectivity to OpenAI API confirmed"
else
  echo "⚠️  Cannot connect to OpenAI API"; fi

echo "Testing connectivity to Ollama..."
if curl -s --max-time 5 "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null; then
  echo "✓ Network connectivity to Ollama confirmed"
else
  echo "⚠️  Cannot connect to Ollama at ${OLLAMA_BASE_URL}"; fi

# Add this after your permissions check section and before running tests

# -------------------------------------------------
# 2.5. Log file structure for debugging
# -------------------------------------------------
echo "Logging file structure for debugging..."
echo "Contents of /app directory:"
find /app -type f -name "*.py" | sort > "${EXPERIMENT_RUN_PATH}/file_structure.log"

# Log Python paths
echo -e "\nPython Path:" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"
python -c "import sys; print('\n'.join(sys.path))" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"

# Check if the specific file exists
echo -e "\nSearching for rstar_data_loader.py:" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"
find /app -name "rstar_data_loader.py" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"

# Print content of directories
echo -e "\nContents of sandbox directory:" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"
ls -la /app/sandbox >> "${EXPERIMENT_RUN_PATH}/file_structure.log" 2>&1

echo -e "\nContents of sandbox/read_only directory:" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"
ls -la /app/sandbox/read_only >> "${EXPERIMENT_RUN_PATH}/file_structure.log" 2>&1

echo -e "\nContents of sandbox/read_only/dataloader directory (if exists):" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"
ls -la /app/sandbox/read_only/dataloader >> "${EXPERIMENT_RUN_PATH}/file_structure.log" 2>&1 || echo "Directory not found" >> "${EXPERIMENT_RUN_PATH}/file_structure.log"

echo "✓ File structure logged to ${EXPERIMENT_RUN_PATH}/file_structure.log"


# -------------------------------------------------
# 3.  Run the infra “setup tests” suite (pytest)
# -------------------------------------------------
echo "Running container setup tests with pytest..."
/app/infra/docker/setup_tests/run_tests.sh
echo "✓ pytest setup tests passed"

# -------------------------------------------------
# 4.  Persist a proof-of-write file in the run dir
# -------------------------------------------------
echo "Verifying writable run directory by leaving a test artefact..."
echo "hello-from-entrypoint $(date -Is)" > "${EXPERIMENT_RUN_PATH}/entrypoint_write_test.txt"
echo "✓ File written to ${EXPERIMENT_RUN_PATH}/entrypoint_write_test.txt"

# -------------------------------------------------
# 5.  (Optional) start the main experiment
# -------------------------------------------------
echo "Executing experiment script with timeout ${EXPERIMENT_TIMEOUT}s ..."
timeout "${EXPERIMENT_TIMEOUT}" python /app/core/scripts/run_experiment.py

# Keep the container alive if no experiment is launched
echo "Container initialisation finished."