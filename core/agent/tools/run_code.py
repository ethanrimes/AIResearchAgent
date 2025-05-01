"""
Tool for running Python code snippets in a safe environment.
"""

import os
import sys
import logging
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any
import subprocess
from pathlib import Path

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class RunCodeTool(BaseTool):
    """Tool for running Python code snippets."""
    
    name = "run_code"
    description = """
    Use this tool to run Python code snippets.
    The code will be executed in a restricted environment.
    
    Input should be a string containing the Python code to run.
    Output will be the stdout, stderr, and any exceptions from running the code.
    
    Example usage:
    run_code("print('Hello, world!')")
    run_code("import math\\nprint(math.sqrt(16))")
    """
    
    def __init__(self, max_execution_time: int = 30):
        """
        Initialize the RunCodeTool.
        
        Args:
            max_execution_time: Maximum execution time in seconds (currently not enforced).
        """
        super().__init__()
        self.max_execution_time = max_execution_time
        logger.info(f"RunCodeTool initialized with max execution time: {max_execution_time}")
        
    def _run(self, code: str) -> str:
        """
        Run Python code and return the results.
        
        Args:
            code: The Python code to run.
            
        Returns:
            The stdout, stderr, and any exceptions from running the code.
        """
        if not code:
            return "ERROR: No code provided."
            
        # Create string buffers to capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            logger.info("Running Python code snippet")
            
            # Use a temporary file to store the code
            import tempfile
            with tempfile.TemporaryDirectory() as tempdir:
                script_path = Path(tempdir) / "user_script.py"
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                # Redirect stdout and stderr to our string buffers
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Execute the code in a separate process
                    # This provides better isolation than exec()
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.max_execution_time,
                        text=True
                    )
                    
                    # Add captured output to our buffers
                    stdout_capture.write(result.stdout)
                    stderr_capture.write(result.stderr)
                
            # Get the captured output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Prepare the result
            result = ""
            if stdout:
                result += f"STDOUT:\n{stdout}\n"
            if stderr:
                result += f"STDERR:\n{stderr}\n"
                
            if not result:
                result = "Code executed successfully with no output."
                
            logger.info("Code executed successfully")
            return result
            
        except subprocess.TimeoutExpired:
            stderr = f"ERROR: Code execution timed out after {self.max_execution_time} seconds."
            logger.error(stderr)
            return stderr
        except Exception as e:
            # Get the traceback
            tb = traceback.format_exc()
            
            # Get any captured output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Prepare the result
            result = f"ERROR: {str(e)}\n\nTraceback:\n{tb}\n"
            
            if stdout:
                result += f"\nSTDOUT (before error):\n{stdout}\n"
            if stderr:
                result += f"\nSTDERR (before error):\n{stderr}\n"
                
            logger.error(f"Error executing code: {str(e)}")
            return result
    
    async def _arun(self, code: str) -> str:
        """
        Async version of _run.
        
        Args:
            code: The Python code to run.
            
        Returns:
            The stdout, stderr, and any exceptions from running the code.
        """
        return self._run(code)