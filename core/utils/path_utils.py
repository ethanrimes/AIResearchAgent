# core/utils/path_utils.py
from pathlib import Path
from typing import Union

def get_active_run_path():
    """
    Get the active run path from /app/active_run_path/path.txt.
    
    Returns:
        Path object representing the active run path.
    """
    from pathlib import Path
    
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