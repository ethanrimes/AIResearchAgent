"""
Implementation of the baseline approach for solving math problems.

This module provides functions for generating baseline prompts and
processing math problems without any reasoning enhancements.
"""

import logging
from typing import Dict, List, Tuple, Any

from sandbox.read_only.models.ollama_client import OllamaClient, ChatMessage, ChatParameters

logger = logging.getLogger(__name__)

class BaselineSolver:
    """
    Solver using the baseline approach for math problems.
    """
    
    def __init__(self, model_name: str = "qwen3:1.7b"):
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
        import time
        
        # Generate the prompt
        prompt = self.generate_prompt(problem)
        
        # Create chat messages
        messages = [
            ChatMessage(role="system", content="You are a mathematical reasoning assistant. Solve problems step-by-step and clearly state your final answer."),
            ChatMessage(role="user", content=prompt)
        ]
        
        # Create chat parameters
        params = ChatParameters(
            model=self.model_name,
            messages=messages,
            stream=False,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1024
            }
        )
        
        # Call the Ollama API and measure time
        start_time = time.time()
        try:
            response = self.ollama_client.chat(params)
            assistant_response = response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error processing problem: {e}")
            assistant_response = f"Error: {str(e)}"
        elapsed_time = time.time() - start_time
        
        return assistant_response, elapsed_time


if __name__ == "__main__":
    # Example usage
    import sys
    import json
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    
    # Sample problem
    sample_problem = {
        "id": "sample1",
        "question": "If x^2 + y^2 = 25 and xy = 12, find x^2 + y^2 + xy.",
        "answer": "37"
    }
    
    # Test the baseline solver
    solver = BaselineSolver()
    response, time_taken = solver.solve(sample_problem)
    
    print(f"Problem: {sample_problem['question']}")
    print(f"Expected answer: {sample_problem['answer']}")
    print(f"Model response (took {time_taken:.2f} seconds):")
    print("-" * 80)
    print(response)
    print("-" * 80)