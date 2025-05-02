"""
Tool for running bash commands in a specified working directory.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunBashInput  # NEW


class RunBashTool(BaseTool):
    """Tool for running bash commands in a specified working directory."""

    name: str = "run_bash"
    description: str = (
        "Run a bash command in the configured working directory. Returns the command "
        "stdout/stderr."
    )

    args_schema = RunBashInput  # NEW

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
            )
            stdout, stderr = proc.communicate(timeout=600)
            rc = proc.returncode

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
            return "ERROR: Command timed out after 600 seconds."
        except Exception as e:
            return f"ERROR: Failed to run command: {str(e)}"

    async def _arun(self, command: str) -> str:
        return self._run(command)