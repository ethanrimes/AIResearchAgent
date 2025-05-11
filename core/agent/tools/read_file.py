"""
Tool for reading files from allowed directories.
"""

import logging
from pathlib import Path
from typing import List
from typing import List, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import ReadFileInput  # NEW


class ReadFileTool(BaseTool):
    """Tool for reading files from allowed directories."""

    name: str = "read_file"
    description: str = (
        "Use this tool to read the contents of a file.\n"
        "You must provide the full path to the file. The file must be located in one "
        "of the allowed directories."
    )
    args_schema: ClassVar[Type[ReadFileInput]] = ReadFileInput


    allowed_dirs: List[Path] = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.read_file"),
        exclude=True,
    )

    def __init__(self, allowed_dirs: List[str]):
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        logger = logging.getLogger("ai_research_agent.read_file")
        super().__init__(allowed_dirs=allowed_dirs_resolved, logger=logger)
        logger.info("ReadFileTool initialised with allowed dirs: %s", allowed_dirs)

    # --------------------------------------------------
    # Helpers
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
    # Core
    # --------------------------------------------------
    def _run(self, file_path: str = None) -> str:  # <- matches schema field name
        # Validate parameters
        if file_path is None:
            return "ERROR: Required parameter 'file_path' is missing"
            
        try:
            path = Path(file_path)

            if not self._is_path_allowed(path):
                allowed_dirs_str = ", ".join(str(d) for d in self.allowed_dirs)
                error_msg = (
                    f"Access denied: {file_path} is not in an allowed directory. "
                    f"Allowed directories: {allowed_dirs_str}"
                )
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            if not path.exists():
                error_msg = f"File not found: {file_path}"
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            if not path.is_file():
                error_msg = f"Not a file: {file_path}"
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.logger.info("Successfully read file: %s", file_path)
            return content
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)