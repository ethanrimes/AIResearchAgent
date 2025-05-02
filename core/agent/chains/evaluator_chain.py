"""
Evaluator Chain module for the AI Research Agent.

This module is responsible for evaluating the results of the experiments
run by the ExperimentChain and producing the final findings.
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

class EvaluatorChain:
    """
    Chain responsible for evaluating experiment results and producing findings.
    """
    
    def __init__(self, llm, workspace: Dict[str, Path], directive: str):
        """
        Initialize the EvaluatorChain.
        
        Args:
            llm: The language model to use.
            workspace: Dictionary containing paths to workspace directories.
            directive: The system directive for the agent.
        """
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.output_file = workspace["findings"] / "final_findings.json"
        self.scratchpad_file = workspace["scratchpad"] / "evaluator_notes.txt"
        
        self.logger = logging.getLogger("ai_research_agent.evaluator_chain")

        # Initialize tools
        self.tools = self._init_tools()
        
        # Initialize agent
        self.agent = self._init_agent()
        
        self.logger.info("EvaluatorChain initialized")
        
    def _init_tools(self) -> List[BaseTool]:
        """
        Initialize the tools for the evaluator agent.
        
        Returns:
            List of tools.
        """
        # For evaluator chain, we can read from read_only, hypothesis, planning, experiment, and results, 
        # but write only to findings
        read_tool = ReadFileTool(
            allowed_dirs=[
                str(self.workspace["read_only"]),
                str(self.workspace["hypothesis"]),
                str(self.workspace["planning"]),
                str(self.workspace["experiment"]),
                str(self.workspace["results"])
            ]
        )
        
        write_tool = WriteFileTool(
            allowed_dirs=[
                str(self.workspace["findings"])
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
        Initialize the evaluator agent.
        
        Returns:
            AgentExecutor instance.
        """
        # Create the prompt template
        prompt = PromptTemplate.from_template(
            template=f"{self.directive}\n\n{{input}}\n\n{{agent_scratchpad}}"
        )
        
        # Create the agent
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10  # Limit iterations to prevent infinite loops
        )
        
    def run(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluation process and produce findings.
        
        Args:
            experiment_results: The results from the ExperimentChain.
            
        Returns:
            Dict containing the final findings.
        """
        self.logger.info("Running result evaluation and generating findings")
        
        hypothesis_file = self.workspace["hypothesis"] / "hypotheses.json"
        plan_file = self.workspace["planning"] / "experiment_plan.json"
        results_file = self.workspace["results"] / "results_summary.json"
        
        # Create minimal input with file paths - directive contains the main instructions
        agent_input = {
            "input": f"Read the hypotheses from: {hypothesis_file}\nRead the experiment plan from: {plan_file}\nRead the experiment results from: {results_file}\nWrite your final findings to: {self.output_file}"
        }
        
        # Run the agent
        result = self.agent.invoke(agent_input)
        
        # Check if the output file was created
        if not self.output_file.exists():
            self.logger.error(f"Evaluator agent failed to create output file: {self.output_file}")
            raise FileNotFoundError(f"Findings output file not found: {self.output_file}")
            
        # Load and return the findings
        try:
            with open(self.output_file, 'r') as f:
                findings = json.load(f)
            self.logger.info("Evaluation completed and findings generated")
            return findings
        except Exception as e:
            self.logger.error(f"Error loading findings output file: {e}")
            raise