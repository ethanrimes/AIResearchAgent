"""
Implementation of the Structured Step-by-Step Decomposition approach for solving math problems.

This module provides functions for generating structured prompts and
processing math problems with explicit step decomposition.
"""

import logging
from typing import Dict, List, Tuple, Any

from sandbox.read_only.models.ollama_client import OllamaClient, ChatMessage, ChatParameters

logger = logging.getLogger(__name__)

class StructuredDecompositionSolver:
    """
    Solver using structured step-by-step decomposition for math problems.
    """
    
    def __init__(self, model_name: str = "qwen3:1.7b"):
        """
        Initialize the structured decomposition solver.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.ollama_client = OllamaClient(default_model=model_name)
        logger.info(f"StructuredDecompositionSolver initialized with model: {model_name}")
    
    def generate_prompt(self, problem: Dict) -> str:
        """
        Generate a structured prompt for a math problem.
        
        Args:
            problem: Dictionary containing the problem
            
        Returns:
            Prompt string
        """
        return f"""Solve this math problem using a structured step-by-step approach:

Step 1: Read the problem carefully and identify what is being asked.
Step 2: Identify the variables and constraints in the problem.
Step 3: Determine the mathematical concepts and formulas needed.
Step 4: Set up equations or a solution strategy.
Step 5: Solve the equations or execute your strategy.
Step 6: Verify your answer makes sense and meets all constraints.
Step 7: State your final answer clearly.

Problem:
{problem['question']}

Begin your structured solution:"""
    
    def solve(self, problem: Dict) -> Tuple[str, float]:
        """
        Solve a math problem using the structured decomposition approach.
        
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
            ChatMessage(role="system", content="You are a mathematical reasoning assistant. Follow the structured approach exactly as outlined and clearly state your final answer."),
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
    
    # Test the structured decomposition solver
    solver = StructuredDecompositionSolver()
    response, time_taken = solver.solve(sample_problem)
    
    print(f"Problem: {sample_problem['question']}")
    print(f"Expected answer: {sample_problem['answer']}")
    print(f"Model response (took {time_taken:.2f} seconds):")
    print("-" * 80)
    print(response)
    print("-" * 80)