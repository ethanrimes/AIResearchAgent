# core.utils.path_utils.py

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