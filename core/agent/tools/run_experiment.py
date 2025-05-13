# core/agent/tools/run_experiment.py

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Type, ClassVar, Tuple, Union

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunExperimentInput
from core.utils.path_utils import get_python_path


class RunExperimentTool(BaseTool):
    """Tool for running experiment scripts in a specified working directory."""

    name: str = "run_experiment"
    description: str = (
        "Run the experiment script in the configured working directory. This tool is "
        "specifically designed to execute the run_experiment.sh script. Returns the command "
        "stdout/stderr."
    )

    args_schema: ClassVar[Type[RunExperimentInput]] = RunExperimentInput

    working_dir: Path = Field(...)
    output_dir: Optional[Path] = None
    timeout: int = Field(default=600)  # Default timeout of 600 seconds if not provided
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.run_experiment"),
        exclude=True,
    )

    def __init__(self, working_dir: str, output_dir: str | None = None, timeout: int = 600):
        working_dir_path = Path(working_dir).resolve()
        output_dir_path = Path(output_dir).resolve() if output_dir else None
        logger = logging.getLogger("ai_research_agent.run_experiment")
        super().__init__(
            working_dir=working_dir_path, output_dir=output_dir_path, 
            timeout=timeout, logger=logger
        )
        working_dir_path.mkdir(parents=True, exist_ok=True)
        if output_dir_path:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized run_experiment tool with timeout: {timeout} seconds")

    def clear_results_directory(self, timestamp=None):
        """
        Clear the results directory by moving files to a timestamped folder in the scratchpad directory.
        
        Args:
            timestamp (str, optional): Timestamp to use for the folder name. If None, current time is used.
        
        Returns:
            str: Message indicating success or failure
        """
        import time
        import shutil
        from glob import glob
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # Define paths
            results_dir = Path("/results")  # Use absolute path
            scratchpad_dir = Path("/scratchpad")  # Use absolute path
            target_dir = scratchpad_dir / f"{timestamp}_results"
            
            # Check if results directory exists
            if not results_dir.exists():
                self.logger.info(f"Results directory does not exist at {results_dir}")
                return f"No results directory found at {results_dir}"
            
            # Ensure scratchpad directory exists
            scratchpad_dir.mkdir(parents=True, exist_ok=True)
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all files in results directory
            result_files = list(results_dir.glob("**/*"))
            files_to_move = [f for f in result_files if f.is_file()]
            
            if not files_to_move:
                self.logger.info(f"No files found in results directory {results_dir}")
                return f"No files found in results directory {results_dir}"
            
            # Move each file preserving directory structure
            for file_path in files_to_move:
                # Get relative path from results directory
                rel_path = file_path.relative_to(results_dir)
                # Create destination path with same structure
                dest_path = target_dir / rel_path
                # Ensure parent directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Move the file
                shutil.move(str(file_path), str(dest_path))
                self.logger.info(f"Moved {file_path} to {dest_path}")
            
            self.logger.info(f"Successfully moved {len(files_to_move)} files to {target_dir}")
            return f"Successfully moved {len(files_to_move)} files from results to {target_dir}"
        
        except Exception as e:
            error_msg = f"Failed to clear results directory: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    def handle_shell_script(self, script_path: str) -> Tuple[bool, Optional[str]]:
        """
        Handle shell script permissions and shebangs.
        
        Args:
            script_path (str): Path to the shell script
            
        Returns:
            Tuple[bool, Optional[str]]: (success, message)
                                    where message is None if no modifications were needed
        """
        import os
        from pathlib import Path
        
        modifications_made = []
        sh_path = Path(script_path)
        
        try:
            # Check if file is executable
            if not os.access(sh_path, os.X_OK):
                os.chmod(sh_path, 0o755)  # Add executable permission
                self.logger.info(f"Added executable permissions to {sh_path}")
                modifications_made.append(f"Added executable permissions to {sh_path}")
            
            # Check if file has a shebang
            with open(sh_path, 'r') as f:
                first_line = f.readline().strip()
                
            if not first_line.startswith('#!'):
                # Add shebang line without overwriting content
                with open(sh_path, 'r') as f:
                    content = f.read()
                
                with open(sh_path, 'w') as f:
                    f.write('#!/bin/bash\n' + content)
                
                self.logger.info(f"Added shebang to {sh_path}")
                modifications_made.append(f"Added shebang line to {script_path}")
        
            message = None
            if modifications_made:
                message = "Script modifications: " + "; ".join(modifications_made)
                
            return True, message
            
        except Exception as e:
            self.logger.error(f"Error handling shell script {sh_path}: {str(e)}")
            return False, f"Failed to modify {script_path}: {str(e)}"

    def _run(self, script_path: str = None, timeout: Optional[int] = None) -> Tuple[str, int]:
        """
        Run an experiment script in the specified working directory.
        
        Args:
            script_path (str): Path to the script to execute, defaults to "run_experiment.sh"
            timeout (Optional[int]): Timeout in seconds for the script execution
            
        Returns:
            Tuple[str, int]: A tuple containing (output_string, return_code)
                            where return_code is 0 for success, non-zero for failure
        """
        # Log all parameters when tool is called
        params_str = ', '.join([f'{k}={v}' for k, v in locals().items() if k != 'self'])
        self.logger.info(f"Tool {self.name} called with parameters: {params_str}")
        
        # Clear results directory first
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clear_result = self.clear_results_directory(timestamp)
        self.logger.info(clear_result)
        
        # Use default script path if not provided
        if script_path is None:
            script_path = "run_experiment.sh"
        
        # Create full path based on working directory
        full_script_path = self.working_dir / script_path
        
        # Validate script exists
        if not full_script_path.exists():
            self.logger.error(f"Script does not exist: {full_script_path}")
            return f"ERROR: Script does not exist: {full_script_path}", 1
        
        # Handle shell script permissions and shebangs
        success, shell_modifications = self.handle_shell_script(full_script_path)
        if not success:
            return f"ERROR: Failed to prepare script: {shell_modifications}", 1
        
        # Use the provided timeout if specified, otherwise use the instance timeout
        effective_timeout = timeout if timeout is not None else self.timeout
        self.logger.info(f"Using timeout of {effective_timeout} seconds for script execution")

        try:
            self.logger.info(f"Running experiment script with timeout {effective_timeout} seconds: {full_script_path}")
            env = os.environ.copy()
            if self.output_dir:
                env["OUTPUT_DIR"] = str(self.output_dir)
            
            # Add Python path
            env["PYTHONPATH"] = get_python_path()
            self.logger.info(f"Set PYTHONPATH environment variable: {env['PYTHONPATH']}")

            proc = subprocess.Popen(
                f"bash {full_script_path}",
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
                self.logger.info(f"Script completed with return code: {rc}")
            except subprocess.TimeoutExpired as e:
                self.logger.error(f"Script timed out after {effective_timeout} seconds")
                proc.kill()
                # Wait for threads to finish after killing the process
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                return f"ERROR: Script timed out after {effective_timeout} seconds.", 1
            
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
                out_file = self.output_dir / f"{ts}_experiment_output.txt"
                self.logger.info(f"Writing script output to {out_file}")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"Script: {full_script_path}\nReturn code: {rc}\n\n")
                    if stdout:
                        f.write(f"STDOUT:\n{stdout}\n")
                    if stderr:
                        f.write(f"STDERR:\n{stderr}\n")

            # Format and return the script result with return code
            if rc == 0:
                if not stdout and not stderr:
                    result = "Script executed successfully with no output."
                    if shell_modifications:
                        result = f"{shell_modifications}\n{result}"
                    self.logger.info(result)
                    return result, rc
                result = ""
                if shell_modifications:
                    result += f"{shell_modifications}\n"
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result, rc
            else:
                self.logger.error(f"Script failed with return code {rc}")
                result = f"Script failed with return code {rc}\n"
                if shell_modifications:
                    result += f"{shell_modifications}\n"
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result, rc
                
        except Exception as e:
            error_msg = f"ERROR: Failed to run script: {str(e)}"
            self.logger.error(error_msg, exc_info=True)  # Include the full traceback
            return error_msg, 1

    async def _arun(self, script_path: str = None, timeout: Optional[int] = None) -> Tuple[str, int]:
        """
        Async version of _run.
        
        Args:
            script_path (str): Path to the script to execute
            timeout (Optional[int]): Timeout in seconds for the script execution
            
        Returns:
            Tuple[str, int]: A tuple containing (output_string, return_code)
        """
        return self._run(script_path, timeout)