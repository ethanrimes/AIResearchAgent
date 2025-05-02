"""
Tool for writing files to allowed directories.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.tools import BaseTool

class WriteFileTool(BaseTool):
    """Tool for writing files to allowed directories."""
    
    name: str = "write_file"
    description: str = """
    Use this tool to write content to a file.
    You must provide the full path to the file and the content to write.
    The file must be in one of the allowed directories.
    This tool will overwrite existing files, so be careful.
    
    Input should be a dictionary with the following keys:
    - file_path: The full path to the file to write.
    - content: The content to write to the file.
    
    Output will be a success or error message.
    """
    
    def __init__(self, allowed_dirs: List[str]):
        """
        Initialize the WriteFileTool.
        
        Args:
            allowed_dirs: List of directories that this tool is allowed to write to.
        """
        super().__init__()
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.logger = logging.getLogger("ai_research_agent.write_file")
        self.logger.info(f"WriteFileTool initialized with allowed dirs: {allowed_dirs}")
        
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
        
    def _run(self, input_dict: Dict[str, str]) -> str:
        """
        Write content to the file at the given path.
        
        Args:
            input_dict: Dictionary containing the file path and content to write.
                Must have keys 'file_path' and 'content'.
            
        Returns:
            Success or error message.
            
        Raises:
            ValueError: If the file path is not allowed or the input is invalid.
        """
        # Validate input
        if not isinstance(input_dict, dict):
            error_msg = f"Invalid input: expected dictionary, got {type(input_dict)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        file_path = input_dict.get('file_path')
        content = input_dict.get('content')
        
        if not file_path:
            error_msg = "Missing required key 'file_path' in input"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        if content is None:  # Allow empty string but not None
            error_msg = "Missing required key 'content' in input"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        path = Path(file_path)
        
        if not self._is_path_allowed(path):
            allowed_dirs_str = ", ".join(str(d) for d in self.allowed_dirs)
            error_msg = f"Access denied: {file_path} is not in an allowed directory. Allowed directories: {allowed_dirs_str}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Successfully wrote to file: {file_path}")
            return f"Successfully wrote to file: {file_path}"
        except Exception as e:
            error_msg = f"Error writing to file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    async def _arun(self, input_dict: Dict[str, str]) -> str:
        """
        Async version of _run.
        
        Args:
            input_dict: Dictionary containing the file path and content to write.
            
        Returns:
            Success or error message.
        """
        return self._run(input_dict)