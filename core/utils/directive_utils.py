"""
Utility functions for loading and processing directive YAML files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from core.utils.path_utils import get_active_run_path


def load_directive(yaml_path: str) -> str:
    """
    Load a directive from a YAML file and convert it to a string format.
    
    Args:
        yaml_path: Path to the YAML file containing the directive.
        
    Returns:
        The directive as a string.
        
    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If the YAML file is invalid.
    """
    logger = logging.getLogger("ai_research_agent.directive_utils")
    try:
        path = Path(yaml_path)
        logger.info(f"Loading directive from {yaml_path}")
        
        if not path.exists():
            error_msg = f"Directive file not found: {yaml_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        with open(path, 'r', encoding='utf-8') as f:
            directive_data = yaml.safe_load(f)
            
        # Convert the directive data to a string format
        return format_directive(directive_data)
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing directive YAML: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading directive: {e}")
        raise
        
def format_directive(directive_data: Dict[str, Any]) -> str:
    """
    Format directive data into a string format.
    
    Args:
        directive_data: Dictionary containing the directive data.
        
    Returns:
        The formatted directive as a string.
    """
    logger = logging.getLogger("ai_research_agent.directive_utils")
    # Get the required fields
    role = directive_data.get('role', 'You are an AI research assistant')
    task = directive_data.get('task', 'Help conduct research on improving LLM reasoning')
    context = directive_data.get('context', '')
    constraints = directive_data.get('constraints', '')
    instructions = directive_data.get('instructions', [])
    output_format = directive_data.get('output_format', '')
    
    # Format the directive
    formatted = f"# Role\n{role}\n\n"
    formatted += f"# Task\n{task}\n\n"
    
    if context:
        formatted += f"# Context\n{context}\n\n"
    
    if constraints:
        formatted += f"# Constraints\n{constraints}\n\n"
        
    if instructions:
        formatted += f"# Instructions\n{instructions}\n\n" 
        
    if output_format:
        formatted += f"# Output Format\n{output_format}\n\n"
    
    # More comprehensive path replacement
    path_mappings = {
        "/app/sandbox/read_only": "/read_only",
        "{ACTIVE_RUN_PATH}/hypothesis": "/hypothesis",
        "{ACTIVE_RUN_PATH}/planning": "/planning",
        "{ACTIVE_RUN_PATH}/experiment": "/experiment",
        "{ACTIVE_RUN_PATH}/results": "/results",
        "{ACTIVE_RUN_PATH}/findings": "/findings",
        "{ACTIVE_RUN_PATH}/scratchpad": "/scratchpad",
        "{ACTIVE_RUN_PATH}": "/",
        "/app/sandbox/writable/run_*/hypothesis": "/hypothesis",
        "/app/sandbox/writable/run_*/planning": "/planning",
        "/app/sandbox/writable/run_*/experiment": "/experiment",
        "/app/sandbox/writable/run_*/results": "/results",
        "/app/sandbox/writable/run_*/findings": "/findings",
        "/app/sandbox/writable/run_*/scratchpad": "/scratchpad"
    }
    
    for old_path, new_path in path_mappings.items():
        formatted = formatted.replace(old_path, new_path)
    
    logger.info(f"Formatted directive with {len(instructions)} instructions")
    return formatted
        
def load_all_directives(directives_path: str) -> Dict[str, str]:
    """
    Load all directives from a directory.
    
    Args:
        directives_path: Path to the directory containing directive YAML files.
        
    Returns:
        Dictionary mapping directive names to formatted directive strings.
    """
    logger = logging.getLogger("ai_research_agent.directive_utils")
    directives = {}
    path = Path(directives_path)
    
    if not path.exists():
        logger.error(f"Directives directory not found: {directives_path}")
        return directives
        
    for yaml_file in path.glob('*.yaml'):
        try:
            directive_name = yaml_file.stem
            directives[directive_name] = load_directive(str(yaml_file))
            logger.info(f"Loaded directive: {directive_name}")
        except Exception as e:
            logger.error(f"Error loading directive {yaml_file}: {e}")
            
    return directives