import os
import sys

# --- Set environment variables for dataloader and model ---
os.environ["DATASET_PATH"] = "data/rstar/aime2024_test.json"
os.environ["OLLAMA_DEFAULT_MODEL"] = "qwen3:0.6b"

# Ensure the core module is on the Python path
sys.path.insert(0, os.path.abspath("."))

from core.agent.tools.run_bash import RunBashTool

# --- Configuration ---
working_dir = "sandbox/writable/run_2025-05-11T14-54-41/experiment"
output_dir = "sandbox/writable/run_2025-05-11T14-54-41/scratchpad"
script_name = "run_experiment.sh"

# --- Instantiate the tool ---
tool = RunBashTool(working_dir=working_dir, output_dir=output_dir)

# --- Construct the command (no need to prepend env vars now) ---
model_name = os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen3:0.6b")
command = f"bash {script_name} {model_name}"

# --- Run the command ---
result = tool._run(command)

# --- Print the result ---
print("=== Command Output ===")
print(result)