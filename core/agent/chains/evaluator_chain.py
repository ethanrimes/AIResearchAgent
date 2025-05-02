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
# WRITE-file tool removed

from core.agent.tools.read_scratchpad import ReadScratchpadTool
from core.agent.tools.write_to_scratchpad import WriteScratchpadTool

from core.agent.schemas.output_schemas import EvaluatorOutput
from core.utils.path_utils import get_active_run_path

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
        # For evaluator chain, we can read from read_only, hypothesis, planning, experiment, and results
        read_tool = ReadFileTool(
            allowed_dirs=[
                str(self.workspace["read_only"]),
                str(self.workspace["hypothesis"]),
                str(self.workspace["planning"]),
                str(self.workspace["experiment"]),
                str(self.workspace["results"])
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
            read_scratchpad_tool,
            write_scratchpad_tool
        ]
        
    def _init_agent(self) -> AgentExecutor:
        """
        Initialize the evaluator agent.
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
        
    def run(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluation process and produce findings.
        
        Args:
            experiment_results: The results from the ExperimentChain.
            
        Returns:
            Dict containing the final findings.
        """
        self.logger.info("Running result evaluation and generating findings")
        
        # ------------------------------------------------------------------
        # Build output path dynamically from the active run directory
        # ------------------------------------------------------------------
        run_dir = get_active_run_path()
        output_path = run_dir / "findings" / "final_findings.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find file paths to reference in the input
        hypothesis_file = self.workspace["hypothesis"] / "hypotheses.json"
        plan_file = self.workspace["planning"] / "experiment_plan.json"
        results_file = self.workspace["results"] / "results_summary.json"
        
        # ------------------------------------------------------------------
        # Structured invocation
        # ------------------------------------------------------------------
        runnable = self.agent.with_structured_output(EvaluatorOutput)
        result: EvaluatorOutput = runnable.invoke(
            {"input": f"Read the hypotheses from: {hypothesis_file}\n"
                      f"Read the experiment plan from: {plan_file}\n"
                      f"Read the experiment results from: {results_file}\n"
                      f"Evaluate these results and return your findings following "
                      f"the required JSON schema."}
        )
        
        # Persist
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(f"Wrote final findings to {output_path}")
        
        return result.model_dump()