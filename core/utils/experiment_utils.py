"""
Utility functions for the experiment phase of the AI Research Agent.

This module provides functions for retrieving experiment code, 
validating experiment outputs, and managing memory during iterations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

def get_current_experiment_code(experiment_dir: Path) -> str:
    """
    Gets the current code in the experiment directory with file annotations.
    
    Args:
        experiment_dir: Path to the experiment directory.
        
    Returns:
        str: A formatted string containing all code files with annotations
    """
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    if not experiment_dir.exists():
        logger.warning(f"Experiment directory does not exist: {experiment_dir}")
        return "No experiment code found."
    
    code_files = list(experiment_dir.glob("**/*.py"))
    if not code_files:
        logger.info("No files found in experiment directory.")
        return "No files found in experiment directory."
    
    result = []
    for file_path in code_files:
        if file_path.is_file():
            try:
                file_content = file_path.read_text()
                file_extension = file_path.suffix[1:] if file_path.suffix else "txt"
                result.append(f"## File: {file_path.name}\n```{file_extension}\n{file_content}\n```\n")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                result.append(f"## File: {file_path.name}\nError reading file: {str(e)}\n")
    
    return "\n".join(result)

def validate_experiment_output(experiment_dir: Path, results_dir: Path, entry_script_name: str = "run_experiment.sh", run_bash_tool=None) -> Dict[str, Any]:
    """
    Validates the experiment output files and directories.
    
    Works with both legacy path structure and flattened structure.
    
    Args:
        experiment_dir: Path to the experiment directory.
        results_dir: Path to the results directory.
        entry_script_name: Name of the entry script file.
        run_bash_tool: Optional RunBashTool to execute the entry script.
        
    Returns:
        Dict containing validation results and errors if any.
    """
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    # Always normalize paths to account for either structure
    if not str(experiment_dir).startswith("/experiment") and Path("/experiment").exists():
        experiment_dir = Path("/experiment")
    
    if not str(results_dir).startswith("/results") and Path("/results").exists():
        results_dir = Path("/results")
        
    # Continue with the existing validation logic using normalized paths
    validation_results = {
        "entry_script_exists": False,
        "python_files_exist": False,
        "results_file_exists": False,
        "bash_output": "",
        "errors": []
    }
    
    # STEP 1: Check that entry script exists and is not empty
    entry_script_path = experiment_dir / entry_script_name
    if not entry_script_path.exists():
        validation_results["errors"].append(f"Entry script does not exist: {entry_script_name}")
        return validation_results
    elif entry_script_path.stat().st_size == 0:
        validation_results["errors"].append(f"Entry script is empty: {entry_script_name}")
        return validation_results
    else:
        validation_results["entry_script_exists"] = True
    
    # STEP 2: Check that Python files exist in experiment directory
    python_files = list(experiment_dir.glob("*.py"))
    if not python_files:
        validation_results["errors"].append("No Python files found in the experiment directory")
        return validation_results
    else:
        validation_results["python_files_exist"] = True
    
    # STEP 3: If we have both entry script and Python files, and run_bash_tool is provided, execute the script
    if run_bash_tool and validation_results["entry_script_exists"] and validation_results["python_files_exist"]:
        try:
            logger.info(f"Executing bash script: {entry_script_path}")
            bash_output = run_bash_tool._run(f"bash {entry_script_path}")
            validation_results["bash_output"] = bash_output
            logger.info(f"Bash script execution completed")
        except Exception as e:
            error_msg = f"Error executing bash script: {str(e)}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
            return validation_results
    
    # STEP 4: After executing the script, check if results_summary.json exists
    results_file_path = results_dir / "results_summary.json"
    if not results_file_path.exists():
        validation_results["errors"].append("Results summary file does not exist: results_summary.json")
    else:
        validation_results["results_file_exists"] = True
    
    logger.info(f"Validation results: {validation_results}")
    return validation_results


def update_memory_with_steps(memory, steps_in_memory, agent_output):
    """
    Updates the agent's memory with new intermediate steps.
    Converts tool actions and observations into SystemMessages.
    
    Args:
        memory: The agent's memory object.
        steps_in_memory: Counter for steps already in memory.
        agent_output: Output from the agent containing intermediate steps.
        
    Returns:
        int: Updated count of steps in memory.
    """
    from langchain.schema import SystemMessage
    
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    if 'intermediate_steps' not in agent_output:
        return steps_in_memory
        
    # Get all steps
    steps = agent_output["intermediate_steps"]
    
    # Determine which ones are new (not yet in memory)
    new_steps = steps[steps_in_memory:]
    
    # Add new steps to memory
    for action, observation in new_steps:
        msg = SystemMessage(
            content=f"Tool '{action.tool}' called with input: {action.tool_input}\n"
                    f"Tool output: {observation}"
        )
        memory.chat_memory.add_message(msg)
    
    # Return updated counter
    updated_steps_count = len(steps)
    logger.info(f"Added {len(new_steps)} new steps to memory. Total steps: {updated_steps_count}")
    
    return updated_steps_count


def run_pyright_type_check(experiment_dir: Path) -> Dict[str, Any]:
    """
    Runs pyright type checking on the Python files in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory.
        
    Returns:
        Dict containing type check results including errors if any.
    """
    import subprocess
    import json
    import os
    from pathlib import Path
    
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    result = {
        "has_errors": False,
        "error_count": 0,
        "errors": []
    }
    
    # Check if pyright is installed
    try:
        # Look for Python files
        python_files = list(experiment_dir.glob("**/*.py"))
        if not python_files:
            logger.info("No Python files found for type checking")
            return result
        
        # Set environment for pyright (use existing environment and update)
        env = os.environ.copy()
        cache_dir = env.get('PYRIGHT_PYTHON_CACHE_DIR')
        if cache_dir:
            logger.info(f"Using pyright cache directory: {cache_dir}")
        
        # Run pyright on each file
        for py_file in python_files:
            try:
                # Use strict mode for better type checking
                cmd = ["pyright", "--outputjson", str(py_file)]
                process = subprocess.run(cmd, capture_output=True, text=True, env=env)
                
                # Parse output
                if process.returncode != 0:
                    # Got errors
                    try:
                        output = json.loads(process.stdout)
                        for diagnostic in output.get("diagnostics", []):
                            if diagnostic.get("severity", "") in ["error", "warning"]:
                                error_message = f"{py_file.name}:{diagnostic.get('line', '?')}: {diagnostic.get('message', 'Unknown error')}"
                                result["errors"].append(error_message)
                                result["error_count"] += 1
                                result["has_errors"] = True
                    except json.JSONDecodeError:
                        # Fallback for when JSON parsing fails
                        error_lines = process.stderr.strip().split("\n")
                        for line in error_lines:
                            if line.strip():
                                result["errors"].append(line.strip())
                                result["error_count"] += 1
                                result["has_errors"] = True
            except Exception as e:
                logger.error(f"Error running pyright on {py_file}: {str(e)}")
                result["errors"].append(f"Failed to check {py_file.name}: {str(e)}")
                result["error_count"] += 1
                result["has_errors"] = True
    
    except FileNotFoundError:
        logger.warning("Pyright not found, skipping type checking")
        result["errors"].append("Pyright not installed - skipping type check")
    except Exception as e:
        logger.error(f"Error during type checking: {str(e)}")
        result["errors"].append(f"Type checking error: {str(e)}")
        result["error_count"] += 1
        result["has_errors"] = True
    
    return result