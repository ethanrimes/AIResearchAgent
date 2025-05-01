"""
Implementation of the baseline approach for solving math problems.

This module provides functions for generating baseline prompts and
processing math problems without any reasoning enhancements.
"""

import logging
import time
from typing import Dict, List, Tuple, Any

from sandbox.read_only.models.ollama_client import OllamaClient, GenerationParameters

logger = logging.getLogger(__name__)

class BaselineSolver:
    """
    Solver using the baseline approach for math problems.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the baseline solver.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.ollama_client = OllamaClient(default_model=model_name)
        logger.info(f"BaselineSolver initialized with model: {model_name}")
    
    def generate_prompt(self, problem: Dict) -> str:
        """
        Generate a baseline prompt for a math problem.
        
        Args:
            problem: Dictionary containing the problem
            
        Returns:
            Prompt string
        """
        return f"Solve this math problem step-by-step:\n\n{problem['question']}"
    
    def solve(self, problem: Dict) -> Tuple[str, float]:
        """
        Solve a math problem using the baseline approach.
        
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