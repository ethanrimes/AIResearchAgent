import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunBashInput  


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
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.run_bash"),
        exclude=True,
    )

    def __init__(self, working_dir: str, output_dir: str | None = None):
        working_dir_path = Path(working_dir).resolve()
        output_dir_path = Path(output_dir).resolve() if output_dir else None
        logger = logging.getLogger("ai_research_agent.run_bash")
        super().__init__(
            working_dir=working_dir_path, output_dir=output_dir_path, logger=logger
        )
        working_dir_path.mkdir(parents=True, exist_ok=True)
        if output_dir_path:
            output_dir_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    def _run(self, command: str) -> str:
        if not command:
            return "ERROR: No command provided."

        try:
            self.logger.info("Running bash command: %s", command)
            env = os.environ.copy()
            if self.output_dir:
                env["OUTPUT_DIR"] = str(self.output_dir)

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
                        self.logger.info(clean_line)
                        stdout_lines.append(line)
                proc.stdout.close()

            def handle_stderr():
                for line in iter(proc.stderr.readline, ''):
                    if line:
                        clean_line = line.rstrip()
                        self.logger.error(clean_line)
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
            rc = proc.wait(timeout=600)
            
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
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"Command: {command}\nReturn code: {rc}\n\n")
                    if stdout:
                        f.write(f"STDOUT:\n{stdout}\n")
                    if stderr:
                        f.write(f"STDERR:\n{stderr}\n")

            # Format and return the command result
            if rc == 0:
                if not stdout and not stderr:
                    return "Command executed successfully with no output."
                result = ""
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result
            else:
                result = f"Command failed with return code {rc}\n"
                if stdout:
                    result += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result += f"STDERR:\n{stderr}\n"
                return result
        except subprocess.TimeoutExpired:
            # Make sure to kill the process if it times out
            proc.kill()
            return "ERROR: Command timed out after 600 seconds."
        except Exception as e:
            return f"ERROR: Failed to run command: {str(e)}"

    async def _arun(self, command: str) -> str:
        return self._run(command)