"""
Tool for updating files with partial modifications.
"""

import logging
from pathlib import Path
from typing import List, Type, ClassVar

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import UpdateFileInput


class UpdateFileTool(BaseTool):
    """Tool for updating files with partial modifications."""

    name: str = "update_file"
    description: str = (
        "Use this tool to update a file with partial modifications. You must provide "
        "the full path to the file, the old content to be replaced, and the new content "
        "to replace it with. The file must be in one of the allowed directories.\n\n"
        "IMPORTANT: The old_content must match EXACTLY with text in the file (including spaces and line breaks). "
        "If old_content appears multiple times, ALL occurrences will be replaced.\n\n"
        "Example: update_file(file_path='/app/config.py', old_content='DEBUG = False', new_content='DEBUG = True')"
    )
    args_schema: ClassVar[Type[UpdateFileInput]] = UpdateFileInput
    allowed_dirs: List[Path] = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.update_file"),
        exclude=True,
    )

    def __init__(self, allowed_dirs: List[str]):
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        logger = logging.getLogger("ai_research_agent.update_file")
        super().__init__(allowed_dirs=allowed_dirs_resolved, logger=logger)
        logger.info("UpdateFileTool initialised with allowed dirs: %s", allowed_dirs)

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
    def _run(self, file_path: str, old_content: str = None, new_content: str = None) -> str:
        # Log all parameters when tool is called
        params_str = f"file_path={file_path}, old_content_length={len(old_content) if old_content else 0}, new_content_length={len(new_content) if new_content else 0}"
        self.logger.info(f"Tool {self.name} called with parameters: {params_str}")
        
        # Validate parameters
        if old_content is None:
            return "ERROR: Required parameter 'old_content' is missing"
        if new_content is None:
            return "ERROR: Required parameter 'new_content' is missing"
        
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

        try:
            # Read the current content
            with open(path, "r", encoding="utf-8") as f:
                current_content = f.read()

            # Check if old_content exists in the file
            if old_content not in current_content:
                error_msg = f"Old content not found in file: {file_path}"
                self.logger.error(error_msg)
                return f"ERROR: {error_msg}"

            # Replace old_content with new_content
            updated_content = current_content.replace(old_content, new_content)

            # Write the updated content back to the file
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            self.logger.info("Successfully updated file: %s", file_path)
            return f"Successfully updated file: {file_path}"
        except Exception as e:
            error_msg = f"Error updating file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"

    async def _arun(self, file_path: str, old_content: str, new_content: str) -> str:
        return self._run(file_path=file_path, old_content=old_content, new_content=new_content)