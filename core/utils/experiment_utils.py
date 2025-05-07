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
    
    code_files = list(experiment_dir.glob("*.*"))
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

def validate_experiment_output(experiment_dir: Path, results_dir: Path, entry_script_name: str = "run_experiment.sh") -> Dict[str, Any]:
    """
    Validates the experiment output files and directories.
    
    Args:
        experiment_dir: Path to the experiment directory.
        results_dir: Path to the results directory.
        entry_script_name: Name of the entry script file.
        
    Returns:
        Dict containing validation results and errors if any.
    """
    logger = logging.getLogger("ai_research_agent.experiment_utils")
    
    validation_results = {
        "entry_script_exists": False,
        "python_files_exist": False,
        "results_file_exists": False,
        "errors": []
    }
    
    # Check 1: Entry script exists and is not empty
    entry_script_path = experiment_dir / entry_script_name
    if not entry_script_path.exists():
        validation_results["errors"].append(f"Entry script does not exist: {entry_script_name}")
    elif entry_script_path.stat().st_size == 0:
        validation_results["errors"].append(f"Entry script is empty: {entry_script_name}")
    else:
        validation_results["entry_script_exists"] = True
    
    # Check 2: Python files exist in experiment directory
    python_files = list(experiment_dir.glob("*.py"))
    if not python_files:
        validation_results["errors"].append("No Python files found in the experiment directory")
    else:
        validation_results["python_files_exist"] = True
    
    # Check 3: Results file exists
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