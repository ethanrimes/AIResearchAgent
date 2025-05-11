"""
Tool for deleting files in allowed directories.
"""

import logging
from pathlib import Path
from typing import List, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import DeleteFileInput


class DeleteFileTool(BaseTool):
    """Tool for deleting files in allowed directories."""

    name: str = "delete_file"
    description: str = (
        "Use this tool to delete a file. You must provide the full path to the file. "
        "The file must be in one of the allowed directories."
    )
    args_schema: ClassVar[Type[DeleteFileInput]] = DeleteFileInput
    allowed_dirs: List[Path] = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.delete_file"),
        exclude=True,
    )

    def __init__(self, allowed_dirs: List[str]):
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        logger = logging.getLogger("ai_research_agent.delete_file")
        super().__init__(allowed_dirs=allowed_dirs_resolved, logger=logger)
        logger.info("DeleteFileTool initialised with allowed dirs: %s", allowed_dirs)

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
    def _run(self, file_path: str = None) -> str:
        # Log all parameters when tool is called
        params_str = ', '.join([f'{k}={v}' for k, v in locals().items() if k != 'self'])
        self.logger.info(f"Tool {self.name} called with parameters: {params_str}")
        
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

            path.unlink()
            self.logger.info("Successfully deleted file: %s", file_path)
            return f"Successfully deleted file: {file_path}"
        except Exception as e:
            error_msg = f"Error deleting file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path=file_path)