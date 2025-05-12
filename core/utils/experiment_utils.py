"""
Utility functions for the experiment phase of the AI Research Agent.

This module provides functions for retrieving experiment code, 
validating experiment outputs, and managing memory during iterations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

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
    python_files = list(experiment_dir.glob("**/*.py"))
    if not python_files:
        validation_results["errors"].append("No Python files found in the experiment directory")
        return validation_results
    else:
        validation_results["python_files_exist"] = True
    
    # STEP 3: If we have both entry script and Python files, and run_bash_tool is provided, execute the script
    # Use the timeout value from the RunBashTool instance if available
    if run_bash_tool and validation_results["entry_script_exists"] and validation_results["python_files_exist"]:
        try:
            logger.info(f"Executing bash script: {entry_script_path}")
            # The RunBashTool._run method now handles the timeout from the tool instance
            bash_output = run_bash_tool._run(command=f"bash {entry_script_path}")
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


def format_experiment_prompt(
    iteration: int,
    directive: str,
    experiment_dir: Path,
    entry_script: str,
    plan_data: Dict[str, Any] = None,
    previous_output: str = None,
    error_information: str = None,
    type_check_results: Dict[str, Any] = None,
    critique_results: Dict[str, Any] = None,
    validation_results: Dict[str, Any] = None
) -> str:
    """
    Format the prompt for the experiment agent based on the current iteration.
    
    Args:
        iteration: Current iteration number.
        directive: The directive for the agent.
        experiment_dir: Path to the experiment directory.
        entry_script: Name of the entry script file.
        plan_data: Optional experiment plan data.
        previous_output: Optional output from previous agent run.
        error_information: Optional error information.
        type_check_results: Optional type check results.
        critique_results: Optional critique results.
        validation_results: Optional validation results.
        
    Returns:
        str: Formatted prompt for the agent.
    """
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    if iteration == 0:
        # Initial prompt for iteration 0
        prompt_parts = [
            f"# Directive\n{directive}\n\n",
            f"# Iteration\n{iteration}\n\n"
        ]
        
        # Add experiment plan if available
        if plan_data:
            experiment_plan = plan_data.get("experiment_plan", {})
            implementation_plan = plan_data.get("implementation_plan", {})
            
            prompt_parts.append(f"# Experiment Plan\n```json\n{json.dumps(experiment_plan, indent=2)}\n```\n\n")
            prompt_parts.append(f"# Implementation Plan\n```json\n{json.dumps(implementation_plan, indent=2)}\n```\n\n")
        
        # Add hypotheses if available
        hypotheses_file = Path("/hypothesis/hypotheses.json")
        if hypotheses_file.exists():
            try:
                hypotheses_content = json.loads(hypotheses_file.read_text())
                prompt_parts.append(f"# Hypotheses\n```json\n{json.dumps(hypotheses_content, indent=2)}\n```\n\n")
            except Exception as e:
                logger.error(f"Error reading hypotheses file: {e}")
        
        # Add API documentation
        api_doc_file = Path("/read_only/api.md")
        if api_doc_file.exists():
            try:
                api_docs = api_doc_file.read_text()
                prompt_parts.append(f"# API Documentation\n# These are the APIs which must be used for data loader, model client, answer extraction and evaluation\n{api_docs}\n\n")
            except Exception as e:
                logger.error(f"Error reading API documentation: {e}")
        
        # Add experiment directory information
        prompt_parts.append(f"# Experiment Code Directory\n{experiment_dir}\n\n")
        
        # Add entry point information
        prompt_parts.append(f"# Entry Point\n{entry_script}\n\n")
        
        # Add instructions for implementation
        prompt_parts.append(
            "# Task\n"
            "Implement the experiment according to the plan. Create all required files in the experiment directory.\n"
            f"Create a bash script called '{entry_script}' that will run the experiment and output results to '/results/results_summary.json'.\n"
            "The experiment must implement all hypotheses and include a baseline calculation for comparison.\n\n"
        )
        
    else:
        # Prompt for subsequent iterations
        prompt_parts = [
            f"# Directive\n{directive}\n\n",
            f"# Iteration\n{iteration}\n\n"
        ]
        
        # Add previous output if available
        if previous_output:
            prompt_parts.append(f"# Previous Output\n{previous_output}\n\n")
        
        # Add current experiment code
        current_code = get_current_experiment_code(experiment_dir)
        prompt_parts.append(f"# Current Experiment Code\n{current_code}\n\n")
        
        # Add error information if available
        if error_information:
            prompt_parts.append(f"# Error Information\n{error_information}\n\n")
        
        # Add type check results if available
        if type_check_results:
            type_check_section = ""
            if type_check_results.get("has_errors", False):
                type_check_section = f"# Type Check Results\nFound {type_check_results.get('error_count', 0)} type errors:\n- " + "\n- ".join(type_check_results.get('errors', [])) + "\n\n"
            else:
                type_check_section = "# Type Check Results\nNo type errors found.\n\n"
            prompt_parts.append(type_check_section)
        
        # Add critique results if available
        if critique_results:
            prompt_parts.append(f"# Critique Results\n```json\n{json.dumps(critique_results, indent=2)}\n```\n\n")
        
        # Add validation results if available
        if validation_results:
            custom_checks = [
                f"- Entry script exists: {validation_results.get('entry_script_exists', False)}",
                f"- Python files in experiment directory: {validation_results.get('python_files_exist', False)}",
                f"- results_summary.json exists: {validation_results.get('results_file_exists', False)}"
            ]
            
            prompt_parts.append(f"# Validation Checks\n" + "\n".join(custom_checks) + "\n\n")
            
            # Add bash output if available
            if validation_results.get('bash_output'):
                prompt_parts.append(f"# Bash Output\n```\n{validation_results.get('bash_output')}\n```\n\n")
        
        # Add task instructions for this iteration
        prompt_parts.append(
            "# Task\n"
            "Review the feedback and continue implementing the experiment. Fix any errors and address the critique.\n"
            f"Ensure that '{entry_script}' correctly runs the experiment and outputs results to '/results/results_summary.json'.\n\n"
        )
    
    return "".join(prompt_parts)

# ------------------------------------------------------------------
# Helper method for logging validation results
# ------------------------------------------------------------------
def log_validation_results(validation_results: Dict[str, Any], iteration: int) -> None:
    """Helper method to log validation results with clear prefacing."""
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    logger.info("=" * 50)
    logger.info(f"VALIDATION RESULTS - Iteration {iteration + 1}")
    logger.info("-" * 50)
    
    # Log core validation checks
    logger.info("VALIDATION: Entry script exists: %s", 
                   "✓ PASS" if validation_results["entry_script_exists"] else "✗ FAIL")
    logger.info("VALIDATION: Python files exist: %s", 
                   "✓ PASS" if validation_results["python_files_exist"] else "✗ FAIL")
    logger.info("VALIDATION: Results file exists: %s", 
                   "✓ PASS" if validation_results["results_file_exists"] else "✗ FAIL")
    
    # Log all errors if any
    if validation_results["errors"]:
        logger.warning("VALIDATION ERRORS:")
        for i, error in enumerate(validation_results["errors"], 1):
            logger.warning("  %d. %s", i, error)
    else:
        logger.info("VALIDATION: No errors detected")
        
    # Log bash output summary if available
    if validation_results.get("bash_output"):
        # Get first few lines of bash output for the log (avoid excessive logging)
        bash_lines = validation_results["bash_output"].split("\n")
        preview = "\n".join(bash_lines[:5])
        if len(bash_lines) > 5:
            preview += f"\n... and {len(bash_lines) - 5} more lines"
        logger.info("VALIDATION: Bash execution output preview:\n%s", preview)
        
    logger.info("VALIDATION: Overall status: %s", 
                   "✓ PASSED" if all([
                       validation_results["entry_script_exists"],
                       validation_results["python_files_exist"], 
                       validation_results["results_file_exists"]
                   ]) else "✗ FAILED")
    logger.info("=" * 50)
