"""
Tool for running Python code snippets in a safe environment.
"""

import logging
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Type, ClassVar


from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import RunCodeInput  # NEW


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

    # --------------------------------------------------
    def _run(self, code: str) -> str:
        if not code:
            return "ERROR: No code provided."

        stdout_buf, stderr_buf = StringIO(), StringIO()

        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "snippet.py"
                script_path.write_text(code, encoding="utf-8")

                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.max_execution_time,
                        text=True,
                    )
                    stdout_buf.write(result.stdout)
                    stderr_buf.write(result.stderr)

            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            if not stdout and not stderr:
                return "Code executed successfully with no output."

            out = ""
            if stdout:
                out += f"STDOUT:\n{stdout}\n"
            if stderr:
                out += f"STDERR:\n{stderr}\n"
            return out
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out after {self.max_execution_time} seconds."
        except Exception as e:
            tb = traceback.format_exc()
            return f"ERROR: {str(e)}\n\nTraceback:\n{tb}"

    async def _arun(self, code: str) -> str:
        return self._run(code)