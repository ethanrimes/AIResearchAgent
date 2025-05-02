"""
Tool for writing to an agent's personal scratch-pad.
"""

import logging
import os
from pathlib import Path

from pydantic import Field
from langchain.tools import BaseTool

from core.agent.schemas.tool_schemas import WriteScratchpadInput  # NEW


class WriteScratchpadTool(BaseTool):
    """Tool for writing to an agent's personal scratch-pad."""

    name: str = "write_to_scratchpad"
    description: str = (
        "Store thoughts, ideas and intermediate results in a personal scratch-pad."
    )
    args_schema = WriteScratchpadInput  # NEW

    scratchpad_file: Path = Field(...)
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(
            "ai_research_agent.write_to_scratchpad"
        ),
        exclude=True,
    )

    def __init__(self, scratchpad_file: str):
        scratchpad_path = Path(scratchpad_file)
        logger = logging.getLogger("ai_research_agent.write_to_scratchpad")
        super().__init__(scratchpad_file=scratchpad_path, logger=logger)
        scratchpad_path.parent.mkdir(parents=True, exist_ok=True)
        scratchpad_path.touch(exist_ok=True)

    # --------------------------------------------------
    def _run(self, content: str, mode: str = "append") -> str:
        try:
            if mode.lower() == "overwrite":
                with open(self.scratchpad_file, "w", encoding="utf-8") as f:
                    f.write(content)
                return "Successfully overwrote scratch-pad."
            else:
                with open(self.scratchpad_file, "a", encoding="utf-8") as f:
                    prefix = "\n" if os.path.getsize(self.scratchpad_file) > 0 else ""
                    f.write(f"{prefix}{content}")
                return "Successfully appended to scratch-pad."
        except Exception as e:
            return f"ERROR: {str(e)}"

    async def _arun(self, content: str, mode: str = "append") -> str:
        return self._run(content=content, mode=mode)