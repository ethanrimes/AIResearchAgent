"""
Tool for reading from an agent's personal scratch-pad.
"""

import logging
from pathlib import Path

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import ReadScratchpadInput  # NEW


class ReadScratchpadTool(BaseTool):
    """Tool for reading from an agent's personal scratch-pad."""

    name: str = "read_scratchpad"
    description: str = "Return the current contents of the personal scratch-pad."
    args_schema = ReadScratchpadInput  # NEW

    scratchpad_file: Path = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("ai_research_agent.read_scratchpad"),
        exclude=True,
    )

    def __init__(self, scratchpad_file: str):
        sp_path = Path(scratchpad_file)
        logger = logging.getLogger("ai_research_agent.read_scratchpad")
        super().__init__(scratchpad_file=sp_path, logger=logger)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.touch(exist_ok=True)

    # --------------------------------------------------
    def _run(self, query: str = "") -> str:
        try:
            content = self.scratchpad_file.read_text(encoding="utf-8")
            return content or "Scratch-pad is empty."
        except Exception as e:
            return f"ERROR: {str(e)}"

    async def _arun(self, query: str = "") -> str:
        return self._run(query)