"""
Evaluator Chain module for the AI Research Agent.

This module is responsible for evaluating the results of the experiments
run by the ExperimentChain and producing the final findings.
"""

import json
import logging
import time  # Added import for timing
from pathlib import Path
from typing import Dict, Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from core.agent.schemas.output_schemas import EvaluatorOutput
from core.utils.path_utils import get_active_run_path
from core.utils.logging_utils import log_agent_invoke  # Added import for logging

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
        
        self.logger = logging.getLogger("ai_research_agent.evaluator_chain")

        # Initialize chain
        self.chain = self._init_chain()
        
        self.logger.info("EvaluatorChain initialized")
        
    def _init_chain(self) -> LLMChain:
        """
        Initialize the LLM chain with the appropriate prompt.
        
        Returns:
            Initialized LLMChain
        """
        prompt = PromptTemplate.from_template(
            template=(
                f"{self.directive}\n\n"
                "# Example Final Findings\n"
                "For reference, here is an example of what well-structured final findings look like:\n"
                "```\n"
                "sandbox/read_only/example/final_findings.json\n"
                "```\n\n"
                "# Generated Hypotheses\n"
                "{hypotheses_content}\n\n"
                "# Experiment Plan\n"
                "{plan_content}\n\n"
                "# Experiment Results\n"
                "{results_content}\n\n"
                "{input}\n\n"
            )
        )

        return LLMChain(
            llm=self.llm,
            prompt=prompt
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
        
        # Start timing the execution
        start_time = time.time()
        
        # ------------------------------------------------------------------
        # Use direct path to findings directory at root level
        # ------------------------------------------------------------------
        output_path = Path("/findings/final_findings.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find file paths and read their contents using direct root paths
        hypothesis_file = Path("/hypothesis/hypotheses.json")
        plan_file = Path("/planning/experiment_plan.json")
        results_file = Path("/results/results_summary.json")
        
        # Read all the necessary files
        try:
            with open(hypothesis_file, 'r', encoding='utf-8') as f:
                hypotheses_content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading hypotheses file: {e}")
            hypotheses_content = "{}"
            
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading plan file: {e}")
            plan_content = "{}"
            
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading results file: {e}")
            results_content = json.dumps(experiment_results, indent=2)
        
        # ------------------------------------------------------------------
        # Structured invocation with the chain
        # ------------------------------------------------------------------
        # Prepare input with all the necessary data embedded
        input_text = "Evaluate the experiment results based on the original hypotheses and experiment plan. " \
                     "Generate comprehensive findings that analyze the effectiveness of each hypothesis, " \
                     "identify successful and unsuccessful reasoning patterns, and suggest future directions. " \
                     "Return your findings following the required JSON schema for EvaluatorOutput."
        
        # Run the chain with structured output
        result = self.llm.with_structured_output(EvaluatorOutput).invoke(
            self.chain.prompt.format(
                hypotheses_content=hypotheses_content,
                plan_content=plan_content,
                results_content=results_content,
                input=input_text
            )
        )
        
        # Persist
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(f"Wrote final findings to {output_path}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log agent invocation details
        log_agent_invoke(
            agent_name="EvaluatorChain",
            runtime=execution_time,
            intermediate_steps=[]  # EvaluatorChain doesn't have intermediate steps to log
        )
        
        return result.model_dump()