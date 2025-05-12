"""
Main agent orchestration module for the AI Research Agent.

This module is responsible for initializing and executing the chain of agents
that will devise, plan, implement, and evaluate experiments to improve LLM reasoning.
"""

import os
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import openai

from core.utils.path_utils import get_active_run_timestamp
from core.agent.agent_config import AgentConfig  # Import the centralized config manager

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import SystemMessage

# Import our custom chains
from core.agent.chains.hypothesis_chain import HypothesisChain
from core.agent.chains.planner_chain import PlannerChain
from core.agent.chains.experiment_chain import ExperimentChain
from core.agent.chains.evaluator_chain import EvaluatorChain
from core.agent.chains.critique_chain import CritiqueChain

# Import utility functions
from core.utils.directive_utils import load_directive

class Agent:
    """
    Main agent class that orchestrates the research process.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the agent with the given configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.logger = logging.getLogger("ai_research_agent.agent")
        
        # Initialize the global configuration registry
        config_registry = AgentConfig()
        self.config = config_registry.load_config(config_path)
        
        # Lock the configuration to prevent further modifications
        config_registry.lock()
        
        try:
            self.run_id = get_active_run_timestamp()
        except Exception:
            # No active run yet – fall back to "now"
            self.run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.workspace = self._setup_workspace()
        
        # Initialize the LLM for the agents
        self.llm = self._init_llm()
        
        # Initialize the agent chains
        self.chains = self._init_chains()
        
        self.logger.info(f"Agent initialized with run ID: {self.run_id}")
        self.logger.info(f"Using agent model: {self.config['agent_model']}")
        self.logger.info(f"Using reasoner model: {self.config['reasoner_model']}")
        self.logger.info(f"Memory window size: {self.config.get('memory_window_size', 0)}")
            
    def _setup_workspace(self) -> Dict[str, Path]:
        """
        Set up the workspace directories for the current run.
        
        Returns:
            Dict containing paths to workspace directories.
        """
        # Get the dataset path from the config registry
        config_registry = AgentConfig()
        dataset_path = config_registry.dataset_path
        
        # Set up workspace using direct root paths
        workspace = {
            "base_dir": Path("/app"),
            "read_only": Path("/read_only"),
            "writable": Path("/"),  # Root as the writable area
            "hypothesis": Path("/hypothesis"),
            "planning": Path("/planning"),
            "experiment": Path("/experiment"),
            "results": Path("/results"),
            "findings": Path("/findings"),
            "scratchpad": Path("/scratchpad"),
            "dataset": Path(dataset_path) if dataset_path else None
        }
        
        # Ensure writable directories exist
        for dir_path in [
            workspace["hypothesis"], 
            workspace["planning"], 
            workspace["experiment"], 
            workspace["results"], 
            workspace["findings"], 
            workspace["scratchpad"]
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.logger.info("Workspace set up with flattened directory structure")
        return workspace
        
    def _init_llm(self) -> ChatOpenAI:
        """
        Initialize the LLM for the agents.
        
        Returns:
            Initialized LLM.
        """
        openai.base_url = "https://api.openai.com/v1"
        self.logger.info(f"Using OpenAI base URL: {openai.base_url}")

        return ChatOpenAI(
            model=self.config["agent_model"],
            # temperature=0.0,  # Use deterministic outputs for research
        )
        
    def _init_chains(self) -> Dict[str, Any]:
        """
        Initialize the agent chains.
        
        Returns:
            Dict containing the initialized chains.
        """
        directives_base_path = self.config["directives_path"]
        
        # Initialize chains in a specific order to ensure dependencies are available
        # First, initialize chains that don't depend on others
        chains = {
            "hypothesis": HypothesisChain(
                llm=self.llm,
                workspace=self.workspace,
                directive=load_directive(f"{directives_base_path}/hypothesis_chain.yaml")
            ),
            "planner": PlannerChain(
                llm=self.llm,
                workspace=self.workspace,
                directive=load_directive(f"{directives_base_path}/planner_chain.yaml")
            ),
            "evaluator": EvaluatorChain(
                llm=self.llm,
                workspace=self.workspace,
                directive=load_directive(f"{directives_base_path}/evaluator_chain.yaml")
            ),
            "critique": CritiqueChain(  # Initialize critique chain before experiment chain
                llm=self.llm,
                workspace=self.workspace,
                directive=load_directive(f"{directives_base_path}/critique_chain.yaml")
            )
        }
        
        # Now initialize experiment chain with critique chain dependency and memory window size
        chains["experiment"] = ExperimentChain(
            llm=self.llm,
            workspace=self.workspace,
            directive=load_directive(f"{directives_base_path}/experiment_chain.yaml"),
            max_iterations=self.config["max_coding_iters"],
            critique_chain=chains["critique"],  # Pass the critique chain
            memory_window_size=self.config.get("memory_window_size", 0)  # Pass memory window size
        )
        
        self.logger.info("Agent chains initialized")
        return chains
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete agent workflow.
        
        Returns:
            Dict containing the results of the run.
        """
        self.logger.info("Starting agent run")
        start_time = time.time()
        max_duration = self.config["max_run_duration"]
        
        try:
            # Step 1: Generate hypotheses
            self.logger.info("Running hypothesis generation")
            hypothesis_result = self.chains["hypothesis"].run()
            
            # Check if we've exceeded the max duration
            if time.time() - start_time > max_duration:
                self.logger.warning("Maximum run duration exceeded after hypothesis generation")
                return {"status": "timeout", "step": "hypothesis"}
                
            # Step 2: Plan the experiment
            self.logger.info("Running experiment planning")
            plan_result = self.chains["planner"].run(hypothesis_result)
            
            # Check if we've exceeded the max duration
            if time.time() - start_time > max_duration:
                self.logger.warning("Maximum run duration exceeded after experiment planning")
                return {"status": "timeout", "step": "planning"}
                
            # Step 3: Implement and run the experiment
            self.logger.info("Running experiment implementation and execution")
            experiment_result = self.chains["experiment"].run(plan_result)
            
            # Check if we've exceeded the max duration
            if time.time() - start_time > max_duration:
                self.logger.warning("Maximum run duration exceeded after experiment execution")
                return {"status": "timeout", "step": "experiment"}
                
            # Step 4: Evaluate the results
            self.logger.info("Running result evaluation")
            evaluation_result = self.chains["evaluator"].run(experiment_result)
            
            # Compile the final results
            elapsed_time = time.time() - start_time
            results = {
                "status": "success",
                "run_id": self.run_id,
                "elapsed_time": elapsed_time,
                "hypothesis": hypothesis_result,
                "plan": plan_result,
                "experiment": experiment_result,
                "evaluation": evaluation_result
            }
            
            self.logger.info(f"Agent run completed successfully in {elapsed_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during agent run: {e}", exc_info=True)
            elapsed_time = time.time() - start_time
            
            return {
                "status": "error",
                "run_id": self.run_id,
                "elapsed_time": elapsed_time,
                "error": str(e)
            }