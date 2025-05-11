#!/bin/bash
set -euo pipefail

# -------------------------------------------------
# 0.  Read environment variables and config file
# -------------------------------------------------
# Extract timeout and reasoner_model from YAML config file with Python
CONFIG_VALUES=$(python -c "
import yaml
with open('/app/core/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
timeout = config.get('max_run_duration', 600)
reasoner_model = config.get('reasoner_model', '')
print(f'{timeout} {reasoner_model}')
")

# Parse the output
read -r CONFIG_TIMEOUT CONFIG_REASONER_MODEL <<< "$CONFIG_VALUES"
echo "CONFIG_TIMEOUT from YAML: ${CONFIG_TIMEOUT}"
echo "CONFIG_REASONER_MODEL from YAML: ${CONFIG_REASONER_MODEL}"

# Make config value the primary source and env var the fallback
EXPERIMENT_TIMEOUT="${CONFIG_TIMEOUT:-${EXPERIMENT_TIMEOUT:-600}}"
echo "Final EXPERIMENT_TIMEOUT being used: ${EXPERIMENT_TIMEOUT}"

# Set OLLAMA_DEFAULT_MODEL from config if not already set
if [ -z "${OLLAMA_DEFAULT_MODEL:-}" ] && [ -n "${CONFIG_REASONER_MODEL}" ]; then
    export OLLAMA_DEFAULT_MODEL="${CONFIG_REASONER_MODEL}"
    echo "Set OLLAMA_DEFAULT_MODEL to ${OLLAMA_DEFAULT_MODEL} from config file"
fi

# Log the final value
echo "Final OLLAMA_DEFAULT_MODEL being used: ${OLLAMA_DEFAULT_MODEL:-not set}"

echo "Using flattened directory structure at root level"
echo "Starting AI Research Agent with experiment timeout: ${EXPERIMENT_TIMEOUT}s"
echo "Using OpenAI API for agent"
echo "Using Ollama for local inference via host.docker.internal"

# Set up log files
LOG_DIR="/logs"
CONTAINER_LOG="${LOG_DIR}/container.log"
FILE_STRUCTURE_LOG="${LOG_DIR}/file_structure.log"

# Ensure log directory exists and is writable
if [[ ! -d "$LOG_DIR" ]]; then
    echo "Creating log directory ${LOG_DIR}..."
    mkdir -p "$LOG_DIR" || { echo "❌ Failed to create ${LOG_DIR}"; exit 1; }
fi

if [[ ! -w "$LOG_DIR" ]]; then
    echo "❌ Cannot write to ${LOG_DIR}"; exit 1
else
    echo "✓ Write access to ${LOG_DIR} confirmed"
fi

# Redirect output to log file and console
exec > >(tee -a "$CONTAINER_LOG") 2>&1
echo "Logging container output to $CONTAINER_LOG"

# -------------------------------------------------
# 1.  Filesystem-permission checks
# -------------------------------------------------
echo "Checking filesystem permissions..."

readonly_ro="/read_only"
readonly_data="/app/data"

if [[ ! -r "$readonly_ro" ]];   then echo "❌ Cannot read $readonly_ro";   exit 1; else echo "✓ Read access to $readonly_ro confirmed"; fi
if [[ ! -r "$readonly_data" ]]; then echo "❌ Cannot read $readonly_data"; exit 1; else echo "✓ Read access to $readonly_data confirmed"; fi

# Set up Pyright cache directory
export PYRIGHT_PYTHON_CACHE_DIR="/scratchpad"
echo "✓ Pyright cache directory set to: ${PYRIGHT_PYTHON_CACHE_DIR}"

# Check permissions on mounted directories
for dir in experiment findings hypothesis planning results logs scratchpad; do
  if [[ ! -w "/$dir" ]]; then
    echo "❌ Cannot write to /$dir"; exit 1
  else
    echo "✓ Write access to /$dir confirmed"
  fi
done

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

# -------------------------------------------------
# 2.5. Log file structure for debugging
# -------------------------------------------------
echo "Logging complete file structure for debugging..."

# First clean previous logs if they exist
> "$FILE_STRUCTURE_LOG"

# Log container ID and hostname
echo "Container ID: $(hostname)" > "$FILE_STRUCTURE_LOG"
echo "Current timestamp: $(date -Is)" >> "$FILE_STRUCTURE_LOG"

# Log environment variables (redacting sensitive values)
echo -e "\n--- Environment Variables ---" >> "$FILE_STRUCTURE_LOG"
env | grep -v "_KEY\|PASSWORD\|TOKEN" | sort >> "$FILE_STRUCTURE_LOG"

# Log mounted volumes
echo -e "\n--- Mounted Volumes ---" >> "$FILE_STRUCTURE_LOG"
mount | grep -E "/experiment|/findings|/hypothesis|/planning|/results|/logs|/scratchpad|/read_only|/app" >> "$FILE_STRUCTURE_LOG"

# Log directory structure
echo -e "\n--- Root Directory Structure ---" >> "$FILE_STRUCTURE_LOG"
ls -la / >> "$FILE_STRUCTURE_LOG" 2>&1

# Log App directory structure - recursively but not too deeply
echo -e "\n--- App Directory Structure ---" >> "$FILE_STRUCTURE_LOG"
find /app -type d -maxdepth 3 | sort >> "$FILE_STRUCTURE_LOG"

# Log Python files specifically
echo -e "\n--- Python Files ---" >> "$FILE_STRUCTURE_LOG"
find /app -type f -name "*.py" | sort >> "$FILE_STRUCTURE_LOG"

# Log Python path
echo -e "\n--- Python Path ---" >> "$FILE_STRUCTURE_LOG"
python -c "import sys; print('\n'.join(sys.path))" >> "$FILE_STRUCTURE_LOG"

# Log key directories content
for dir in /app/core /app/infra /experiment /findings /hypothesis /planning /results /logs /scratchpad /read_only; do
  if [ -d "$dir" ]; then
    echo -e "\n--- Contents of $dir directory ---" >> "$FILE_STRUCTURE_LOG"
    ls -la "$dir" >> "$FILE_STRUCTURE_LOG" 2>&1
  fi
done

# Log disk space
echo -e "\n--- Disk Space ---" >> "$FILE_STRUCTURE_LOG"
df -h >> "$FILE_STRUCTURE_LOG"

echo "✓ Complete file structure logged to $FILE_STRUCTURE_LOG"

# -------------------------------------------------
# 3.  Run the infra "setup tests" suite (pytest)
# -------------------------------------------------
echo "Running container setup tests with pytest..."
/app/infra/docker/setup_tests/run_tests.sh
echo "✓ pytest setup tests passed"

# -------------------------------------------------
# 4.  Persist a proof-of-write file
# -------------------------------------------------
echo "Verifying writable directories by leaving a test artefact..."
echo "hello-from-entrypoint $(date -Is)" > "/logs/entrypoint_write_test.txt"
echo "✓ File written to /logs/entrypoint_write_test.txt"

# -------------------------------------------------
# 5.  (Optional) start the main experiment
# -------------------------------------------------
echo "Executing experiment script with timeout ${EXPERIMENT_TIMEOUT}s ..."
timeout "${EXPERIMENT_TIMEOUT}" python /app/core/scripts/run_experiment.py

# Keep the container alive if no experiment is launched
echo "Container initialisation finished."