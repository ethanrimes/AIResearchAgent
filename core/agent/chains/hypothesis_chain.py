"""
Hypothesis Chain module for the AI Research Agent.

This module is responsible for generating hypotheses about ways to improve
LLM reasoning capacity, particularly for mathematical reasoning tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from core.agent.schemas.output_schemas import HypothesisOutput
from core.utils.path_utils import get_active_run_path

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
        
        # Initialize logger as a property
        self.logger = logging.getLogger("ai_research_agent.hypothesis_chain")
        
        # Initialize the chain
        self.chain = self._init_chain()
        
        self.logger.info("HypothesisChain initialized")
        
    def _init_chain(self) -> LLMChain:
        """
        Initialize the LLM chain with the appropriate prompt.
        
        Returns:
            Initialized LLMChain
        """
        prompt = PromptTemplate.from_template(
            template=(
                f"{self.directive}\n\n"
                "# Example Hypotheses\n"
                "For reference, here is an example of what well-structured hypotheses look like:\n"
                "```\n"
                "sandbox/read_only/example/hypotheses.json\n"
                "```\n\n"
                "{input}\n\n"
            )
        )

        return LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
    def run(self) -> Dict[str, Any]:
        """
        Run the hypothesis generation process.
        
        Returns:
            Dict containing the generated hypotheses.
        """
        self.logger.info("Running hypothesis generation")
        
        # ------------------------------------------------------------------
        # Build output path dynamically from the active run directory
        # ------------------------------------------------------------------
        run_dir = get_active_run_path()
        output_path = run_dir / "hypothesis" / "hypotheses.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Structured invocation with the chain
        # ------------------------------------------------------------------
        # Prepare input with any example data embedded
        input_text = "Generate your hypotheses about ways to improve LLM reasoning capabilities. " \
                     "Focus on mathematical reasoning tasks in particular. " \
                     "Return them following the required JSON schema for HypothesisOutput."
        
        # Run the chain with structured output
        result = self.llm.with_structured_output(HypothesisOutput).invoke(
            self.chain.prompt.format(input=input_text)
        )

        # Persist the result
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(f"Wrote hypotheses to {output_path}")

        return result.model_dump()