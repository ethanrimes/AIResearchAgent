"""
Tool for reading from an agent's personal scratchpad.
"""

import os
import logging
from pathlib import Path
from pydantic import Field

from langchain.tools import BaseTool


class ReadScratchpadTool(BaseTool):
    """Tool for reading from an agent's personal scratchpad."""
    
    name: str = "read_scratchpad"
    description: str = """
    Use this tool to read the contents of your personal scratchpad.
    You can use this to retrieve your previous thoughts and notes.
    
    Input should be an empty string or the word "read".
    Output will be the contents of your scratchpad.
    """
    scratchpad_file: Path = Field(...)
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(
        "ai_research_agent.read_scratchpad"), exclude=True)
    
    def __init__(self, scratchpad_file: str):
        """
        Initialize the ReadScratchpadTool.
        
        Args:
            scratchpad_file: Path to the scratchpad file.
        """
        scratchpad_file_path = Path(scratchpad_file)
        logger = logging.getLogger("ai_research_agent.read_scratchpad")
        super().__init__(scratchpad_file=scratchpad_file_path, logger=logger)
        logger.info(f"ReadScratchpadTool initialized with scratchpad file: {scratchpad_file}")

        # Create the file if it doesn't exist
        scratchpad_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not scratchpad_file_path.exists():
            with open(scratchpad_file_path, 'w', encoding='utf-8') as f:
                f.write("")
        
    def _run(self, query: str = "") -> str:
        """
        Read the contents of the scratchpad.
        
        Args:
            query: Ignored. Can be empty or any string.
            
        Returns:
            Contents of the scratchpad as a string.
        """
        try:
            if not self.scratchpad_file.exists():
                self.logger.info(f"Scratchpad file not found, creating empty file: {self.scratchpad_file}")
                with open(self.scratchpad_file, 'w', encoding='utf-8') as f:
                    f.write("")
                return "Scratchpad is empty."
                
            with open(self.scratchpad_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content:
                return "Scratchpad is empty."
                
            self.logger.info(f"Successfully read scratchpad")
            return content
        except Exception as e:
            error_msg = f"Error reading scratchpad: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    async def _arun(self, query: str = "") -> str:
        """
        Async version of _run.
        
        Args:
            query: Ignored. Can be empty or any string.
            
        Returns:
            Contents of the scratchpad as a string.
        """
        return self._run(query)