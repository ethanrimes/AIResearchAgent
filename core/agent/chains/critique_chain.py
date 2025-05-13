"""
Critique Chain module for the AI Research Agent.

This module is responsible for reviewing experiment code against requirements
and providing structured feedback without using any external tools.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from core.agent.schemas.output_schemas import CritiqueOutput
from core.utils.path_utils import get_active_run_path
from core.utils.experiment_utils import get_current_experiment_code, run_pyright_type_check


class CritiqueChain:
    """
    Chain responsible for reviewing experiment code against requirements.
    """
    
    def __init__(self, llm, workspace: Dict[str, Path], directive: str):
        """
        Initialize the CritiqueChain.
        
        Args:
            llm: The language model to use.
            workspace: Dictionary containing paths to workspace directories.
            directive: The system directive for the agent.
        """
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.output_file = workspace["scratchpad"] / "critique_results.json"
        
        self.logger = logging.getLogger("ai_research_agent.critique_chain")
        
        # Initialize the chain
        self.chain = self._init_chain()
        
        self.logger.info("CritiqueChain initialized")
        
    def _init_chain(self) -> LLMChain:
        """
        Initialize the LLM chain with the appropriate prompt.
        
        Returns:
            Initialized LLMChain
        """
        prompt = PromptTemplate.from_template(
            template=(
                f"{self.directive}\n\n"
                "# Code to Review\n"
                "{experiment_code}\n\n"
                "# Experiment Plan\n"
                "{experiment_plan}\n\n"
                "# Generated Hypotheses\n"
                "{hypotheses_content}\n\n"
                "{input}\n\n"
            )
        )

        return LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
    def run(self, experiment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the critique process to review experiment code.
        
        Args:
            experiment_data: Optional additional experiment data to include in the review.
            
        Returns:
            Dict containing the critique results.
        """
        self.logger.info("Running experiment code critique")
        
        # ------------------------------------------------------------------
        # Use direct path to scratchpad directory at root level
        # ------------------------------------------------------------------
        # Generate timestamp for unique filename
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"/scratchpad/{ts}_critique_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current experiment code using direct path
        experiment_code = get_current_experiment_code(Path("/experiment"))
        
        # Find the hypothesis and plan files to read from root paths
        hypothesis_file = Path("/hypothesis/hypotheses.json")
        plan_file = Path("/planning/experiment_plan.json")
        
        # Get type check results on direct experiment path
        type_check_results = run_pyright_type_check(Path("/experiment"))
        
        hypotheses_content = "{}"
        experiment_plan = "{}"
        
        if hypothesis_file.exists():
            try:
                hypotheses_content = json.dumps(json.loads(hypothesis_file.read_text()), indent=2)
            except Exception as e:
                self.logger.error(f"Error reading hypothesis file: {e}")
                
        if plan_file.exists():
            try:
                experiment_plan = json.dumps(json.loads(plan_file.read_text()), indent=2)
            except Exception as e:
                self.logger.error(f"Error reading plan file: {e}")
        
        # ------------------------------------------------------------------
        # Structured invocation with the chain
        # ------------------------------------------------------------------
        # Prepare input
        input_text = ("Review the experiment implementation code against the specified requirements "
                    "and output a structured evaluation. Focus on validating that all hypotheses "
                    "are properly implemented, all required files exist, and the code meets "
                    "all requirements in the checklist. Evaluate progress against each step in the "
                    "build sequence from the experiment plan. Include type checking results in your evaluation.")
        
        # Run the chain with structured output
        result = self.llm.with_structured_output(CritiqueOutput).invoke(
            self.chain.prompt.format(
                experiment_code=experiment_code,
                experiment_plan=experiment_plan,
                hypotheses_content=hypotheses_content,
                type_check_results=json.dumps(type_check_results, indent=2),
                input=input_text
            )
        )
        
        # Persist results
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(f"Wrote critique results to {output_path}")
        
        return result.model_dump()