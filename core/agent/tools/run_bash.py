# core/agent/tools/run_bash.py

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunBashInput
from core.utils.path_utils import get_python_path


class RunBashTool(BaseTool):
    """Tool for running bash commands in a specified working directory."""

    name: str = "run_bash"
    description: str = (
        "Run a bash command in the configured working directory. Returns the command "
        "stdout/stderr."
    )

    args_schema: ClassVar[Type[RunBashInput]] = RunBashInput

    working_dir: Path = Field(...)
    output_dir: Optional[Path] = None
    timeout: int = Field(default=600)  # Default timeout of 600 seconds if not provided
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.run_bash"),
        exclude=True,
    )

    def __init__(self, working_dir: str, output_dir: str | None = None, timeout: int = 600):
        working_dir_path = Path(working_dir).resolve()
        output_dir_path = Path(output_dir).resolve() if output_dir else None
        logger = logging.getLogger("ai_research_agent.run_bash")
        super().__init__(
            working_dir=working_dir_path, output_dir=output_dir_path, 
            timeout=timeout, logger=logger
        )
        working_dir_path.mkdir(parents=True, exist_ok=True)
        if output_dir_path:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized run_bash tool with timeout: {timeout} seconds")

    # --------------------------------------------------
    def _run(self, command: str = None, timeout: Optional[int] = None) -> str:
        # Log all parameters when tool is called
        params_str = ', '.join([f'{k}={v}' for k, v in locals().items() if k != 'self'])
        self.logger.info(f"Tool {self.name} called with parameters: {params_str}")
        
        # Validate parameters
        if command is None:
            self.logger.error("Required parameter 'command' is missing")
            return "ERROR: Required parameter 'command' is missing"
            
        # Use the provided timeout if specified, otherwise use the instance timeout
        effective_timeout = timeout if timeout is not None else self.timeout
        self.logger.info(f"Using timeout of {effective_timeout} seconds for command execution")

        try:
            self.logger.info(f"Running bash command with timeout {effective_timeout} seconds: {command}")
            env = os.environ.copy()
            if self.output_dir:
                env["OUTPUT_DIR"] = str(self.output_dir)
            
            # Add Python path
            env["PYTHONPATH"] = get_python_path()
            self.logger.info(f"Set PYTHONPATH environment variable: {env['PYTHONPATH']}")

            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
                text=True,
                bufsize=1  # Line-buffered
            )

            # Collect output in these lists
            stdout_lines = []
            stderr_lines = []

            # Define thread functions for handling stdout and stderr
            def handle_stdout():
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        clean_line = line.rstrip()
                        # Always log each line of stdout at INFO level
                        self.logger.info("[STDOUT] %s", clean_line)
                        stdout_lines.append(line)
                proc.stdout.close()

            def handle_stderr():
                for line in iter(proc.stderr.readline, ''):
                    if line:
                        clean_line = line.rstrip()
                        # Always log each line of stderr at WARNING level
                        self.logger.warning("[STDERR] %s", clean_line)
                        stderr_lines.append(line)
                proc.stderr.close()

            # Start threads to handle stdout and stderr in real-time
            stdout_thread = threading.Thread(target=handle_stdout)
            stderr_thread = threading.Thread(target=handle_stderr)
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete with timeout
            try:
                rc = proc.wait(timeout=effective_timeout)
                self.logger.info(f"Command completed with return code: {rc}")
            except subprocess.TimeoutExpired as e:
                self.logger.error(f"Command timed out after {effective_timeout} seconds")
                proc.kill()
                # Wait for threads to finish after killing the process
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                return f"ERROR: Command timed out after {effective_timeout} seconds."
            
            # Wait for output threads to finish
            stdout_thread.join()
            stderr_thread.join()

            # Combine collected output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            # Persist output if requested
            if self.output_dir:
                import time
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_file = self.output_dir / f"bash_output_{ts}.txt"
                self.logger.info(f"Writing command output to {out_file}")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"Command: {command}\nReturn code: {rc}\n\n")
                    if stdout:
                        f.write(f"STDOUT:\n{stdout}\n")
                    if stderr:
                        f.write(f"STDERR:\n{stderr}\n")

            # Format and return the command result
            if rc == 0:
                if not stdout and not stderr:
                    self.logger.info("Command executed successfully with no output")
                    return "Command executed successfully with no output."
                result = ""
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result
            else:
                self.logger.error(f"Command failed with return code {rc}")
                result = f"Command failed with return code {rc}\n"
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result
                
        except Exception as e:
            error_msg = f"ERROR: Failed to run command: {str(e)}"
            self.logger.error(error_msg, exc_info=True)  # Include the full traceback
            return error_msg

    async def _arun(self, command: str, timeout: Optional[int] = None) -> str:
        return self._run(command, timeout)