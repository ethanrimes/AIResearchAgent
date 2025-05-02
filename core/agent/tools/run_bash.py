"""
Tool for running bash commands in a specified working directory.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field

from langchain.tools import BaseTool

class RunBashTool(BaseTool):
    """Tool for running bash commands in a specified working directory."""
    
    name: str = "run_bash"
    description: str = """
    Use this tool to run bash commands.
    You must provide the command to run.
    The command will be run in the specified working directory.
    The output will be directed to the specified output directory.
    
    Input should be a string containing the bash command to run.
    Output will be the stdout and stderr from the command.
    
    Example usage:
    run_bash("ls -la")
    run_bash("python experiment.py")
    """
    working_dir: Path = Field(...)
    output_dir: Optional[Path] = None
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(
        "ai_research_agent.run_bash"), exclude=True)

    def __init__(self, working_dir: str, output_dir: str = None):
        """
        Initialize the RunBashTool.
        
        Args:
            working_dir: Directory to run commands in.
            output_dir: Directory to write command output to. If not provided, 
                       output will only be returned to the caller.
        """
        super().__init__()
        self.working_dir = Path(working_dir).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.logger = logging.getLogger("ai_research_agent.run_bash")
        self.logger.info(f"RunBashTool initialized with working directory: {working_dir}")
        if output_dir:
            self.logger.info(f"RunBashTool output directory: {output_dir}")
        
        # Create the directories if they don't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            self.logger.info(f"Running bash command: {command}")
            
            # Set environment variables to restrict output
            env = os.environ.copy()
            if self.output_dir:
                env["OUTPUT_DIR"] = str(self.output_dir)
            
            # Run the command in the specified working directory
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
                text=True
            )
            
            # Get stdout and stderr
            stdout, stderr = process.communicate(timeout=600)  # 10-minute timeout
            return_code = process.returncode
            
            # Also write output to file if output_dir is specified
            if self.output_dir:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = self.output_dir / f"bash_output_{timestamp}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Command: {command}\n")
                    f.write(f"Return code: {return_code}\n\n")
                    if stdout:
                        f.write(f"STDOUT:\n{stdout}\n")
                    if stderr:
                        f.write(f"STDERR:\n{stderr}\n")
            
            if return_code == 0:
                self.logger.info(f"Command executed successfully with return code {return_code}")
                if not stdout and not stderr:
                    return "Command executed successfully with no output."
                output = ""
                if stdout:
                    output += f"STDOUT:\n{stdout}\n"
                if stderr:
                    output += f"STDERR:\n{stderr}\n"
                return output
            else:
                self.logger.warning(f"Command failed with return code {return_code}")
                output = f"Command failed with return code {return_code}\n"
                if stdout:
                    output += f"STDOUT:\n{stdout}\n"
                if stderr:
                    output += f"STDERR:\n{stderr}\n"
                return output
                
        except subprocess.TimeoutExpired:
            self.logger.error("Command timed out after 600 seconds")
            return "ERROR: Command timed out after 600 seconds."
        except Exception as e:
            self.logger.error(f"Error running command: {str(e)}")
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