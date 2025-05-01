"""
Tool for reading files from allowed directories.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.tools import BaseTool

class ReadFileTool(BaseTool):
    """Tool for reading files from allowed directories."""
    
    name = "read_file"
    description = """
    Use this tool to read the contents of a file.
    You must provide the full path to the file.
    The file must be in one of the allowed directories.
    
    Input should be a single string containing the file path.
    Output will be the file contents as a string.
    """
    
    def __init__(self, allowed_dirs: List[str]):
        """
        Initialize the ReadFileTool.
        
        Args:
            allowed_dirs: List of directories that this tool is allowed to read from.
        """
        super().__init__()
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.logger = logging.getLogger("ai_research_agent.read_file")
        self.logger.info(f"ReadFileTool initialized with allowed dirs: {allowed_dirs}")
        
    def _is_path_allowed(self, file_path: Path) -> bool:
        """
        Check if the given file path is in one of the allowed directories.
        
        Args:
            file_path: Path to check.
            
        Returns:
            True if the path is allowed, False otherwise.
        """
        resolved_path = file_path.resolve()
        
        # Check if the path is in any of the allowed directories
        for allowed_dir in self.allowed_dirs:
            try:
                # Use relative_to to check if the path is within the allowed directory
                # This will raise ValueError if not
                resolved_path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
                
        return False
        
    def _run(self, file_path: str) -> str:
        """
        Read the contents of the file at the given path.
        
        Args:
            file_path: Path to the file to read.
            
        Returns:
            Contents of the file as a string.
            
        Raises:
            ValueError: If the file path is not allowed or the file doesn't exist.
        """
        path = Path(file_path)
        
        if not self._is_path_allowed(path):
            allowed_dirs_str = ", ".join(str(d) for d in self.allowed_dirs)
            error_msg = f"Access denied: {file_path} is not in an allowed directory. Allowed directories: {allowed_dirs_str}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        if not path.exists():
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        if not path.is_file():
            error_msg = f"Not a file: {file_path}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"Successfully read file: {file_path}")
            return content
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    async def _arun(self, file_path: str) -> str:
        """
        Async version of _run.
        
        Args:
            file_path: Path to the file to read.
            
        Returns:
            Contents of the file as a string.
        """
        return self._run(file_path)