"""
Tool for writing files to allowed directories.
"""

import logging
from pathlib import Path
from typing import List, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import WriteFileInput  # NEW


class WriteFileTool(BaseTool):
    """Tool for writing files to allowed directories."""

    name: str = "write_file"
    description: str = (
        "Use this tool to write content to a file. You must provide the full path to "
        "the file and the content to write. The file must be in one of the allowed "
        "directories. This tool overwrites existing files."
    )
    args_schema: ClassVar[Type[WriteFileInput]] = WriteFileInput
    allowed_dirs: List[Path] = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.write_file"),
        exclude=True,
    )

    def __init__(self, allowed_dirs: List[str]):
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        logger = logging.getLogger("ai_research_agent.write_file")
        super().__init__(allowed_dirs=allowed_dirs_resolved, logger=logger)
        logger.info("WriteFileTool initialised with allowed dirs: %s", allowed_dirs)

    # --------------------------------------------------
    def _is_path_allowed(self, file_path: Path) -> bool:
        resolved_path = file_path.resolve()
        for allowed_dir in self.allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    # --------------------------------------------------
    def _run(self, file_path: str = None, content: str = None) -> str:
        # Validate parameters
        if file_path is None:
            return "ERROR: Required parameter 'file_path' is missing"
        if content is None:
            return "ERROR: Required parameter 'content' is missing"
        
        
        path = Path(file_path)

        if not self._is_path_allowed(path):
            allowed_dirs_str = ", ".join(str(d) for d in self.allowed_dirs)
            error_msg = (
                f"Access denied: {file_path} is not in an allowed directory. "
                f"Allowed directories: {allowed_dirs_str}"
            )
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"

        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info("Successfully wrote to file: %s", file_path)
            return f"Successfully wrote to file: {file_path}"
        except Exception as e:
            error_msg = f"Error writing to file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"

    async def _arun(self, file_path: str, content: str) -> str:
        return self._run(file_path=file_path, content=content)