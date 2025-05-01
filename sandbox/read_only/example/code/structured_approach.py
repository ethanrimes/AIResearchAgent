"""
Implementation of the Structured Step-by-Step approach for solving math problems.

This module provides functions for generating structured prompts and
processing math problems with a structured reasoning technique.
"""

import logging
import time
from typing import Dict, List, Tuple, Any

from sandbox.read_only.models.ollama_client import OllamaClient, GenerationParameters

logger = logging.getLogger(__name__)

class StructuredSolver:
    """
    Solver using structured step-by-step reasoning for math problems.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the structured solver.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.ollama_client = OllamaClient(default_model=model_name)
        logger.info(f"StructuredSolver initialized with model: {model_name}")
    
    def generate_prompt(self, problem: Dict) -> str:
        """
        Generate a structured prompt for a math problem.
        
        Args:
            problem: Dictionary containing the problem
            
        Returns:
            Prompt string
        """
        return f"""Solve this math problem using a structured approach:

Step 1: Read the problem carefully and identify what is being asked.
Step 2: Identify the variables and constraints in the problem.
Step 3: Determine the mathematical concepts and formulas needed.
Step 4: Set up equations or a solution strategy.
Step 5: Solve the equations or execute your strategy.
Step 6: Verify your answer meets all constraints.
Step 7: State your final answer clearly.

Problem:
{problem['question']}

Begin your structured solution:"""
    
    def solve(self, problem: Dict) -> Tuple[str, float]:
        """
        Solve a math problem using the structured approach.
        
        Args:
            problem: Dictionary containing the problem
            
        Returns:
            Tuple of (model_response, execution_time)
        """
        # Generate the prompt
        prompt = self.generate_prompt(problem)
        
        # Create generation parameters
        params = GenerationParameters(
            model=self.model_name,  # Will be overridden by OllamaClient
            prompt=prompt,
            temperature=0.1,
            top_p=0.9
        )
        
        # Call the Ollama API and measure time
        start_time = time.time()
        try:
            response = self.ollama_client.generate(params)
            model_output = response.get("response", "")
        except Exception as e:
            logger.error(f"Error processing problem: {e}")
            model_output = f"Error: {str(e)}"
        elapsed_time = time.time() - start_time
        
        return model_output, elapsed_time