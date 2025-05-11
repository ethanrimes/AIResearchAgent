"""
Tool for listing files in allowed directories.
"""

import logging
from pathlib import Path
from typing import List, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import ListFilesInput


class ListFilesTool(BaseTool):
    """Tool for listing files in allowed directories."""

    name: str = "list_files"
    description: str = (
        "Use this tool to list files in a directory. You must provide the full path "
        "to the directory. The directory must be one of the allowed directories or a subdirectory."
    )
    args_schema: ClassVar[Type[ListFilesInput]] = ListFilesInput
    allowed_dirs: List[Path] = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.list_files"),
        exclude=True,
    )

    def __init__(self, allowed_dirs: List[str]):
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        logger = logging.getLogger("ai_research_agent.list_files")
        super().__init__(allowed_dirs=allowed_dirs_resolved, logger=logger)
        logger.info("ListFilesTool initialised with allowed dirs: %s", allowed_dirs)

    # --------------------------------------------------
    def _is_path_allowed(self, directory_path: Path) -> bool:
        resolved_path = directory_path.resolve()
        for allowed_dir in self.allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir)
                return True
            except ValueError:
                # Check if the allowed directory is relative to the requested directory
                try:
                    allowed_dir.relative_to(resolved_path)
                    return True
                except ValueError:
                    continue
        return False

    # --------------------------------------------------
    def _run(self, directory_path: str = None) -> str:
    # Validate parameters
        if directory_path is None:
            return "ERROR: Required parameter 'directory_path' is missing"
        
        try:
            path = Path(directory_path)

            if not self._is_path_allowed(path):
                allowed_dirs_str = ", ".join(str(d) for d in self.allowed_dirs)
                error_msg = (
                    f"Access denied: {directory_path} is not in an allowed directory. "
                    f"Allowed directories: {allowed_dirs_str}"
                )
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            if not path.exists():
                error_msg = f"Directory not found: {directory_path}"
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            if not path.is_dir():
                error_msg = f"Not a directory: {directory_path}"
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            # List all files in the directory
            files = [str(f.relative_to(path)) for f in path.iterdir() if f.is_file()]
            dirs = [str(d.relative_to(path)) + "/" for d in path.iterdir() if d.is_dir()]
            
            entries = sorted(dirs) + sorted(files)
            
            if not entries:
                return f"Directory is empty: {directory_path}"
                
            # Format the response
            result = f"Contents of {directory_path}:\n"
            for entry in entries:
                result += f"- {entry}\n"
                
            self.logger.info("Successfully listed files in: %s", directory_path)
            return result
        except Exception as e:
            error_msg = f"Error listing files in {directory_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"

    async def _arun(self, directory_path: str) -> str:
        return self._run(directory_path=directory_path)