"""
Implementation of the Structured Step-by-Step approach for solving math problems.

This module provides a solver that uses a structured reasoning approach
to guide the model through a step-by-step process.
"""

import logging
import time
from typing import Dict, Tuple

from sandbox.read_only.models.ollama_client import OllamaClient, GenerationParameters

logger = logging.getLogger(__name__)

class StructuredSolver:
    """
    Solver using structured step-by-step reasoning for math problems.
    """
    
    def __init__(self):
        """
        Initialize the structured solver.
        
        The model to use is handled by the OllamaClient through environment variables.
        """
        self.ollama_client = OllamaClient()
        logger.info(f"StructuredSolver initialized with model: {self.ollama_client.default_model}")
    
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
            model="",  # This will be ignored by OllamaClient
            prompt=prompt,
            temperature=0.1,  # Low temperature for more deterministic responses
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