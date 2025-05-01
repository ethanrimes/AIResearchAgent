"""
Tool for running bash commands in a specified working directory.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class RunBashTool(BaseTool):
    """Tool for running bash commands in a specified working directory."""
    
    name = "run_bash"
    description = """
    Use this tool to run bash commands.
    You must provide the command to run.
    The command will be run in the specified working directory.
    
    Input should be a string containing the bash command to run.
    Output will be the stdout and stderr from the command.
    
    Example usage:
    run_bash("ls -la")
    run_bash("python experiment.py")
    """
    
    def __init__(self, working_dir: str):
        """
        Initialize the RunBashTool.
        
        Args:
            working_dir: Directory to run commands in.
        """
        super().__init__()
        self.working_dir = Path(working_dir).resolve()
        logger.info(f"RunBashTool initialized with working directory: {working_dir}")
        
        # Create the directory if it doesn't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
    def _run(self, command: str) -> str:
        """
        Run a bash command in the specified working directory.
        
        Args:
            command: The bash command to run.
            
        Returns:
            The stdout and stderr from the command.
        """
        if not command:
            return "ERROR: No command provided."
            
        try:
            logger.info(f"Running bash command: {command}")
            
            # Run the command in the specified working directory
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                text=True
            )
            
            # Get stdout and stderr
            stdout, stderr = process.communicate(timeout=600)  # 10-minute timeout
            return_code = process.returncode
            
            if return_code == 0:
                logger.info(f"Command executed successfully with return code {return_code}")
                if not stdout and not stderr:
                    return "Command executed successfully with no output."
                output = ""
                if stdout:
                    output += f"STDOUT:\n{stdout}\n"
                if stderr:
                    output += f"STDERR:\n{stderr}\n"
                return output
            else:
                logger.warning(f"Command failed with return code {return_code}")
                output = f"Command failed with return code {return_code}\n"
                if stdout:
                    output += f"STDOUT:\n{stdout}\n"
                if stderr:
                    output += f"STDERR:\n{stderr}\n"
                return output
                
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after 600 seconds")
            return "ERROR: Command timed out after 600 seconds."
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return f"ERROR: Failed to run command: {str(e)}"
    
    async def _arun(self, command: str) -> str:
        """
        Async version of _run.
        
        Args:
            command: The bash command to run.
            
        Returns:
            The stdout and stderr from the command.
        """
        return self._run(command)