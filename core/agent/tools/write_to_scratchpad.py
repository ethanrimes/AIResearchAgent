"""
Tool for writing to an agent's personal scratchpad.
"""

import os
import logging
from pathlib import Path
from pydantic import Field

from langchain.tools import BaseTool

class WriteScratchpadTool(BaseTool):
    """Tool for writing to an agent's personal scratchpad."""
    
    name: str = "write_to_scratchpad"
    description: str = """
    Use this tool to write content to your personal scratchpad.
    You can use this to store your thoughts, ideas, and intermediate results.
    
    Input should be the content to write to the scratchpad.
    You can choose to append or overwrite the existing content by specifying
    'mode' as either 'append' or 'overwrite'. Default is 'append'.
    
    For append mode:
    {
        "content": "Your content here",
        "mode": "append"
    }
    
    For overwrite mode:
    {
        "content": "Your content here",
        "mode": "overwrite"
    }
    
    Output will be a success or error message.
    """
    scratchpad_file: Path = Field(...)
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(
        "ai_research_agent.write_to_scratchpad"), exclude=True)
    
    def __init__(self, scratchpad_file: str):
        """
        Initialize the WriteScratchpadTool.
        
        Args:
            scratchpad_file: Path to the scratchpad file.
        """
        super().__init__()
        self.scratchpad_file = Path(scratchpad_file)
        self.logger = logging.getLogger("ai_research_agent.write_to_scratchpad")
        self.logger.info(f"WriteScratchpadTool initialized with scratchpad file: {scratchpad_file}")
        
        # Create the file if it doesn't exist
        self.scratchpad_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.scratchpad_file.exists():
            with open(self.scratchpad_file, 'w', encoding='utf-8') as f:
                f.write("")
        
    def _run(self, input_dict) -> str:
        """
        Write content to the scratchpad.
        
        Args:
            input_dict: Dictionary containing the content to write and the mode.
                If a string is provided, it will be treated as content in append mode.
            
        Returns:
            Success or error message.
        """
        try:
            # Handle both string and dictionary inputs
            if isinstance(input_dict, str):
                content = input_dict
                mode = "append"
            else:
                content = input_dict.get('content', '')
                mode = input_dict.get('mode', 'append')
                
            # Create parent directories if they don't exist
            self.scratchpad_file.parent.mkdir(parents=True, exist_ok=True)
            
            if mode.lower() == 'overwrite':
                with open(self.scratchpad_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Successfully overwrote scratchpad")
                return "Successfully overwrote scratchpad."
            else:  # Default to append mode
                with open(self.scratchpad_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{content}" if os.path.getsize(self.scratchpad_file) > 0 else content)
                self.logger.info(f"Successfully appended to scratchpad")
                return "Successfully appended to scratchpad."
        except Exception as e:
            error_msg = f"Error writing to scratchpad: {str(e)}"
            self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    async def _arun(self, input_dict) -> str:
        """
        Async version of _run.
        
        Args:
            input_dict: Dictionary containing the content to write and the mode.
            
        Returns:
            Success or error message.
        """
        return self._run(input_dict)