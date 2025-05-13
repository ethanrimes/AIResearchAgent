"""
Logging utilities for the AI Research Agent.

This module provides functions for setting up and configuring logging.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from core.utils.path_utils import get_active_run_timestamp
from typing import Dict, Any, Optional


def setup_logging(run_dir=None, level=logging.INFO, log_name=None, capture_docker=True):
    """
    Set up logging with console and file handlers.
    
    Args:
        run_dir (str, optional): Directory where logs should be stored. 
                                 If None, logs will only be output to console.
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_name (str, optional): Name for the log file. 
                                 If None, a timestamped name will be used.
        capture_docker (bool): Whether to capture Docker logs. Defaults to True.
                                 
    Returns:
        logging.Logger: Configured logger object.
    """
    # Format for log messages
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create root logger
    root_logger = logging.getLogger('ai_research_agent')
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplication
    while root_logger.handlers:
        root_logger.handlers.pop()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if run_dir is provided
    if run_dir:
        # Create logs directory if it doesn't exist
        log_dir = Path(run_dir) # / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename
        if log_name is None:
            try:
                timestamp = get_active_run_timestamp()
            except Exception:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_name = f"experiment_logs_{timestamp}.log"
        
        # Create file handler
        log_file_path = log_dir / log_name
        file_handler = logging.FileHandler(str(log_file_path))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Log file created at: {log_file_path}")
    
    # Make sure handlers propagate to child loggers
    root_logger.propagate = False
    
    # Log initialization message
    root_logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    
    return root_logger

def get_logger(name=None):
    """
    Get a logger with the given name.
    
    Args:
        name (str, optional): Name for the logger. If None, returns the root ai_research_agent logger.
        
    Returns:
        logging.Logger: Logger object.
    """
    if not name:
        return logging.getLogger('ai_research_agent')
    
    # For child loggers, use the full namespace
    return logging.getLogger(f'ai_research_agent.{name}')


def log_validation_results(validation_results: Dict[str, Any], enriched_checklist: Optional[Dict[str, Any]] = None, iteration: int = 0) -> None:
    """
    Log validation results with clear prefacing.
    
    Args:
        validation_results: Dictionary containing validation results.
        enriched_checklist: Optional enriched checklist from process_validation_checklist.
        iteration: Current iteration number.
    """
    logger = logging.getLogger("ai_research_agent.logging_utils")
    
    logger.info("=" * 50)
    logger.info(f"VALIDATION RESULTS - Iteration {iteration + 1}")
    logger.info("-" * 50)
    
    # Log core validation checks
    logger.info("VALIDATION: Entry script exists: %s", 
                   "✓ PASS" if validation_results.get("entry_script_exists", False) else "✗ FAIL")
    logger.info("VALIDATION: Python files exist: %s", 
                   "✓ PASS" if validation_results.get("python_files_exist", False) else "✗ FAIL")
    logger.info("VALIDATION: Results file exists: %s", 
                   "✓ PASS" if validation_results.get("results_file_exists", False) else "✗ FAIL")
    
    # Log enriched validation if available
    if enriched_checklist:
        logger.info("-" * 50)
        logger.info("DETAILED VALIDATION CHECKS:")
        
        # Log regular requirements
        for key, value in enriched_checklist.items():
            if key not in ["entry_script_exists", "python_files_exist", "results_file_exists"]:
                if isinstance(value, dict) and "is_met" in value:
                    status = "✓ PASS" if value["is_met"] else "✗ FAIL"
                    logger.info("VALIDATION: %s: %s", value.get("description", key), status)
                    if value.get("comment") and not value["is_met"]:
                        logger.info("   Comment: %s", value["comment"])
                elif key == "hypotheses_implementation":
                    # Special handling for hypotheses
                    all_implemented = value.get("all_implemented", False)
                    status = "✓ PASS" if all_implemented else "✗ FAIL"
                    logger.info("VALIDATION: %s: %s", value.get("description", key), status)
                    
                    # Log individual hypothesis details
                    if "implementation_details" in value:
                        for detail in value["implementation_details"]:
                            hyp_status = "✓ PASS" if detail.get("is_implemented", False) else "✗ FAIL"
                            logger.info("   - Hypothesis %s: %s", detail.get("hypothesis_id", "?"), hyp_status)
                            if detail.get("comment") and not detail.get("is_implemented", False):
                                logger.info("     Comment: %s", detail["comment"])
    
    # Log all errors if any
    if validation_results.get("errors"):
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
    
    # Determine overall status based on enriched checklist if available
    if enriched_checklist:
        all_passed = True
        for key, value in enriched_checklist.items():
            if isinstance(value, dict) and "is_met" in value:
                if not value["is_met"]:
                    all_passed = False
                    break
            elif key == "hypotheses_implementation":
                if not value.get("all_implemented", False):
                    all_passed = False
                    break
        
        logger.info("VALIDATION: Overall status: %s", 
                       "✓ PASSED" if all_passed else "✗ FAILED")
    else:
        # Fall back to basic validation
        logger.info("VALIDATION: Basic validation status: %s", 
                       "✓ PASSED" if all([
                           validation_results.get("entry_script_exists", False),
                           validation_results.get("python_files_exist", False), 
                           validation_results.get("results_file_exists", False)
                       ]) else "✗ FAILED")
    
    logger.info("=" * 50)