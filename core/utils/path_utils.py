# core/utils/path_utils.py
from pathlib import Path
from typing import Union
import logging
import sys

def get_active_run_path():
    """
    Get the active run path.
    
    In the flattened directory structure, this returns the root path.
    Legacy code may still call this function, so we maintain it.
    
    Returns:
        Path object representing the root path.
    """
    logger = logging.getLogger("ai_research_agent.path_utils")
    
    # Check if we're using the flattened structure
    if Path("/experiment").exists() and Path("/findings").exists():
        logger.info("Using flattened directory structure with root paths")
        return Path("/")
    
    # Fall back to the legacy path file method
    active_run_path_file = Path("/app/active_run_path/path.txt")

    if not active_run_path_file.exists():
        raise FileNotFoundError("Active run path file not found: /app/active_run_path/path.txt")

    with open(active_run_path_file, 'r') as f:
        run_path = f.read().strip()

    if not run_path:
        raise ValueError("Active run path is empty")

    run_path = Path(run_path)

    if not run_path.exists():
        run_path.mkdir(parents=True, exist_ok=True)

    return run_path



def _extract_timestamp_from_run_name(run_name: str) -> str:
    """
    Given a directory name like ``run_2025-05-02T11-05-37`` return
    ``2025-05-02T11-05-37``.
    """
    if not run_name.startswith("run_"):
        raise ValueError(f"Run name does not start with 'run_': {run_name}")
    ts = run_name[4:]
    if not ts:
        raise ValueError("Timestamp segment is empty")
    return ts


def get_run_timestamp_from_path(path: Union[str, Path]) -> str:
    """
    Search the supplied path for the first segment that matches ``run_*`` and
    return its timestamp portion.

    Args:
        path: Path object (or str) that contains a ``run_<timestamp>`` segment.

    Returns:
        str: The extracted timestamp (e.g. ``2025-05-02T11-05-37``).
    """
    p = Path(path)
    for part in reversed(p.parts):
        if part.startswith("run_"):
            return _extract_timestamp_from_run_name(part)

    raise ValueError(f"No 'run_<timestamp>' segment found in path: {path}")


def get_active_run_timestamp() -> str:
    """
    Convenience wrapper that pulls the active run path via
    ``get_active_run_path`` and returns its timestamp.

    Returns:
        str: Active run timestamp.
    """
    return get_run_timestamp_from_path(get_active_run_path())


def get_python_path() -> str:
    """
    Returns a properly formatted PYTHONPATH environment variable string that includes
    the standard directories needed for AI Research Agent tools.
    
    Returns:
        str: PYTHONPATH string with experiment and read_only directories
    """
    logger = logging.getLogger("ai_research_agent.path_utils")
    
    # Start with the current Python path
    current_path = ":".join(sys.path)
    
    # Standard directories to include
    standard_dirs = ["/experiment", "/read_only"]
    
    # Check which directories exist
    dirs_to_add = [dir_path for dir_path in standard_dirs if Path(dir_path).exists()]
    
    if dirs_to_add:
        logger.info(f"Adding directories to PYTHONPATH: {', '.join(dirs_to_add)}")
        # Create a path string with the directories that exist
        path_addition = ":".join(dirs_to_add)
        
        # Combine with current path if it exists
        if current_path:
            python_path = f"{path_addition}:{current_path}"
        else:
            python_path = path_addition
            
        logger.info(f"Configured PYTHONPATH: {python_path}")
        return python_path
    else:
        logger.warning("No standard directories found to add to PYTHONPATH")
        return current_path