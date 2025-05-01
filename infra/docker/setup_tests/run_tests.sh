# infra/docker/setup_tests/run_tests.sh
#!/usr/bin/env bash
set -euo pipefail

# run pytest against this folder
pytest -q --disable-warnings --maxfail=1 "$(dirname "$0")"
echo "✔ all setup tests passed"