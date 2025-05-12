#!/usr/bin/env bash
set -euo pipefail

# Change directory to the setup tests folder
cd "$(dirname "$0")"

# Set up logging
echo "Running tests with verbose output: $@"

# Run pytest with verbose output against the specified test files
# "$@" passes all arguments from the command line to pytest
python -m pytest -v --no-cache-dir "$@"

echo "✓ all setup tests passed"