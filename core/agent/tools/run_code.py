# core/agent/tools/run_code.py

"""
Tool for running Python code snippets in a safe environment.
"""

import logging
import subprocess
import sys
import traceback
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunCodeInput
from core.utils.path_utils import get_python_path


class RunCodeTool(BaseTool):
    """Tool for running Python code snippets."""

    name: str = "run_code"
    description: str = "Execute a Python code snippet in an isolated subprocess."
    args_schema: ClassVar[Type[RunCodeInput]] = RunCodeInput

    max_execution_time: int = Field(default=30)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.run_code"),
        exclude=True,
    )

    def __init__(self, max_execution_time: int = 30):
        logger = logging.getLogger("ai_research_agent.run_code")
        super().__init__(max_execution_time=max_execution_time, logger=logger)
        logger.info("RunCodeTool initialized with max_execution_time: %s", max_execution_time)

    # --------------------------------------------------
    def _run(self, code: str = None) -> str:
        # Log all parameters when tool is called
        self.logger.info(f"Tool {self.name} called with code snippet of length: {len(code) if code else 0}")
        
        # Validate parameters
        if code is None:
            return "ERROR: Required parameter 'code' is missing"
        
        try:
            self.logger.info("Executing Python code snippet")
            stdout_buf, stderr_buf = StringIO(), StringIO()

            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "snippet.py"
                script_path.write_text(code, encoding="utf-8")
                self.logger.info(f"Created temporary script at {script_path}")

                # Set up environment with PYTHONPATH
                env = os.environ.copy()
                env["PYTHONPATH"] = get_python_path()
                self.logger.info(f"Set PYTHONPATH environment variable: {env['PYTHONPATH']}")

                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.max_execution_time,
                        text=True,
                        env=env,
                    )
                    stdout_buf.write(result.stdout)
                    stderr_buf.write(result.stderr)
                    self.logger.info(f"Code execution completed with return code: {result.returncode}")

            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            if stdout:
                self.logger.info(f"Code execution produced stdout: {stdout[:500]}{'...' if len(stdout) > 500 else ''}")
            if stderr:
                self.logger.error(f"Code execution produced stderr: {stderr}")

            if not stdout and not stderr:
                self.logger.info("Code executed successfully with no output")
                return "Code executed successfully with no output."

            out = ""
            if stdout:
                out += f"STDOUT:\n{stdout}\n"
            if stderr:
                out += f"STDERR:\n{stderr}\n"
            return out
        except subprocess.TimeoutExpired:
            error_msg = f"ERROR: Code execution timed out after {self.max_execution_time} seconds."
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error running code: {str(e)}\n{tb}")
            return f"ERROR: {str(e)}\n\nTraceback:\n{tb}"

    async def _arun(self, code: str) -> str:
        return self._run(code)