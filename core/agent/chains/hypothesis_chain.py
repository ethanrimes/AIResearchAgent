"""
Hypothesis Chain module for the AI Research Agent.

This module is responsible for generating hypotheses about ways to improve
LLM reasoning capacity, particularly for mathematical reasoning tasks.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_structured_chat_agent

# Import custom tools
from core.agent.tools.read_file import ReadFileTool
from core.agent.tools.write_file import WriteFileTool
from core.agent.tools.read_scratchpad import ReadScratchpadTool
from core.agent.tools.write_to_scratchpad import WriteScratchpadTool

class HypothesisChain:
    """
    Chain responsible for generating hypotheses about ways to improve LLM reasoning.
    """
    
    def __init__(self, llm, workspace: Dict[str, Path], directive: str):
        """
        Initialize the HypothesisChain.
        
        Args:
            llm: The language model to use.
            workspace: Dictionary containing paths to workspace directories.
            directive: The system directive for the agent.
        """
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.output_file = workspace["hypothesis"] / "hypotheses.json"
        self.scratchpad_file = workspace["scratchpad"] / "hypothesis_notes.txt"
        
        # Initialize logger as a property
        self.logger = logging.getLogger("ai_research_agent.hypothesis_chain")
        
        # Initialize tools
        self.tools = self._init_tools()
        
        # Initialize agent
        self.agent = self._init_agent()
        
        self.logger.info("HypothesisChain initialized")
        
    def _init_tools(self) -> List[BaseTool]:
        """
        Initialize the tools for the hypothesis agent.
        
        Returns:
            List of tools.
        """
        # For hypothesis chain, we can read from read_only and write only to hypothesis
        read_tool = ReadFileTool(
            allowed_dirs=[
                str(self.workspace["read_only"])
            ]
        )
        
        write_tool = WriteFileTool(
            allowed_dirs=[
                str(self.workspace["hypothesis"])
            ]
        )
        
        read_scratchpad_tool = ReadScratchpadTool(
            scratchpad_file=str(self.scratchpad_file)
        )
        
        write_scratchpad_tool = WriteScratchpadTool(
            scratchpad_file=str(self.scratchpad_file)
        )
        
        return [
            read_tool,
            write_tool,
            read_scratchpad_tool,
            write_scratchpad_tool
        ]
        
    def _init_agent(self) -> AgentExecutor:
        """
        Initialize the hypothesis agent.
        """
        prompt = PromptTemplate.from_template(
            template=(
                f"{self.directive}\n\n"
                "Tool names you can call: {tool_names}\n\n"
                "Full tool descriptions:\n{tools}\n\n"
                "{input}\n\n"
                "{agent_scratchpad}"
            )
        )

        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            stop_sequence=False
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
        
    def run(self) -> Dict[str, Any]:
        """
        Run the hypothesis generation process.
        
        Returns:
            Dict containing the generated hypotheses.
        """
        self.logger.info("Running hypothesis generation")
        
        # Create input with output file path - directive contains the main instructions
        agent_input = {
            "input": f"Your final hypotheses should be written to: {self.output_file}"
        }
        
        # Run the agent
        result = self.agent.invoke(agent_input)
        
        # Check if the output file was created
        if not self.output_file.exists():
            self.logger.error(f"Hypothesis agent failed to create output file: {self.output_file}")
            raise FileNotFoundError(f"Hypothesis output file not found: {self.output_file}")
            
        # Load and return the generated hypotheses
        try:
            with open(self.output_file, 'r') as f:
                hypotheses = json.load(f)
            self.logger.info(f"Generated {len(hypotheses['hypotheses'])} hypotheses")
            return hypotheses
        except Exception as e:
            self.logger.error(f"Error loading hypothesis output file: {e}")
            raise