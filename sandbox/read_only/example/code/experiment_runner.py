"""
Main experiment runner for testing LLM reasoning enhancement techniques.

This script loads math problems from a dataset, tests different reasoning techniques,
and evaluates the results.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add parent directories to path so we can import the modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.append('/app')

# Import data loader and evaluation tools
from sandbox.read_only.dataloader.rstar_data_loader import rStarDataLoader
from sandbox.read_only.evaluator.answer_evaluation import AnswerEvaluator
from sandbox.read_only.models.ollama_client import OllamaClient, ChatMessage, ChatParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Class to run experiments testing different reasoning enhancement techniques.
    """
    
    def __init__(self, dataset_path: str, output_dir: str, model_name: str = "qwen3:1.7b"):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_path: Path to the dataset file
            output_dir: Directory to write results
            model_name: Name of the Ollama model to use
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = rStarDataLoader(dataset_path, logger)
        self.evaluator = AnswerEvaluator()
        self.ollama_client = OllamaClient(default_model=model_name)
        
        # Check if Ollama model is available
        self._check_model_availability()
        
        logger.info(f"ExperimentRunner initialized with model: {model_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")
    
    def _check_model_availability(self):
        """Check if the specified model is available in Ollama."""
        try:
            models = self.ollama_client.list_models()
            available_models = [model["name"] for model in models]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                logger.warning("The experiment might fail when trying to use this model.")
            else:
                logger.info(f"Model {self.model_name} is available.")
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
    
    def run_baseline(self, problems: List[Dict], max_problems: int = 10) -> Dict:
        """
        Run the baseline approach without enhancement techniques.
        
        Args:
            problems: List of problem dictionaries
            max_problems: Maximum number of problems to process
            
        Returns:
            Dictionary with baseline results
        """
        logger.info(f"Running baseline on up to {max_problems} problems")
        
        # Limit the number of problems for testing
        test_problems = problems[:max_problems]
        
        results = {
            "accuracy": 0.0,
            "examples": [],
            "summary": {
                "total_problems": len(test_problems),
                "correct_answers": 0,
                "accuracy": 0.0,
                "average_time": 0.0
            }
        }
        
        total_time = 0.0
        correct_count = 0
        
        for i, problem in enumerate(test_problems):
            logger.info(f"Processing baseline problem {i+1}/{len(test_problems)}: {problem['id']}")
            
            # Construct the prompt
            prompt = f"Solve this math problem step-by-step:\n\n{problem['question']}"
            
            # Process the problem
            start_time = time.time()
            response = self._process_problem(prompt)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate(response, problem['answer'])
            
            if is_correct:
                correct_count += 1
            
            # Record the result
            results["examples"].append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Calculate summary statistics
        if test_problems:
            results["summary"]["correct_answers"] = correct_count
            results["summary"]["accuracy"] = round(correct_count / len(test_problems), 2)
            results["summary"]["average_time"] = round(total_time / len(test_problems), 2)
        
        results["accuracy"] = results["summary"]["accuracy"]
        
        logger.info(f"Baseline completed with accuracy: {results['accuracy']}")
        return results
    
    def run_structured_decomposition(self, problems: List[Dict], max_problems: int = 10) -> Dict:
        """
        Run the structured step-by-step decomposition technique.
        
        Args:
            problems: List of problem dictionaries
            max_problems: Maximum number of problems to process
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Running structured decomposition on up to {max_problems} problems")
        
        # Limit the number of problems for testing
        test_problems = problems[:max_problems]
        
        results = {
            "id": "H1",
            "accuracy": 0.0,
            "examples": [],
            "summary": {
                "total_problems": len(test_problems),
                "correct_answers": 0,
                "accuracy": 0.0,
                "average_time": 0.0
            }
        }
        
        total_time = 0.0
        correct_count = 0
        
        for i, problem in enumerate(test_problems):
            logger.info(f"Processing structured decomposition problem {i+1}/{len(test_problems)}: {problem['id']}")
            
            # Construct the enhanced prompt
            prompt = f"""Solve this math problem using a structured step-by-step approach:

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
            
            # Process the problem
            start_time = time.time()
            response = self._process_problem(prompt)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate(response, problem['answer'])
            
            if is_correct:
                correct_count += 1
            
            # Record the result
            results["examples"].append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Calculate summary statistics
        if test_problems:
            results["summary"]["correct_answers"] = correct_count
            results["summary"]["accuracy"] = round(correct_count / len(test_problems), 2)
            results["summary"]["average_time"] = round(total_time / len(test_problems), 2)
        
        results["accuracy"] = results["summary"]["accuracy"]
        
        logger.info(f"Structured decomposition completed with accuracy: {results['accuracy']}")
        return results
    
    def _process_problem(self, prompt: str) -> str:
        """
        Process a math problem using the Ollama model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response
        """
        try:
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
                    "temperature": 0.1,  # Low temperature for more deterministic responses
                    "top_p": 0.9,
                    "max_tokens": 1024   # Enough tokens for a detailed solution
                }
            )
            
            # Call the Ollama API
            response = self.ollama_client.chat(params)
            
            # Extract the assistant's message
            assistant_response = response.get("message", {}).get("content", "")
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error processing problem: {e}")
            return f"Error: {str(e)}"
    
    def run_experiment(self, max_problems: int = 10) -> Dict:
        """
        Run the full experiment testing baseline and all hypotheses.
        
        Args:
            max_problems: Maximum number of problems to process per technique
            
        Returns:
            Dictionary with complete experiment results
        """
        logger.info("Starting full experiment")
        
        # Load the data
        problems = self.data_loader.load()
        
        if not problems:
            logger.error("No problems loaded from dataset")
            return {"error": "No problems loaded from dataset"}
        
        logger.info(f"Loaded {len(problems)} problems from dataset")
        
        # Run baseline
        baseline_results = self.run_baseline(problems, max_problems)
        
        # Run hypothesis H1: Structured Step-by-Step Decomposition
        h1_results = self.run_structured_decomposition(problems, max_problems)
        
        # Compile the results
        results = {
            "experiment_results": {
                "baseline": baseline_results,
                "hypotheses": [h1_results],
                "comparison": {
                    "best_performing": "Baseline" if baseline_results["accuracy"] >= h1_results["accuracy"] else "H1",
                    "accuracy_improvements": {
                        "H1": round(h1_results["accuracy"] - baseline_results["accuracy"], 2)
                    }
                }
            },
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset": self.dataset_path,
                "model": self.model_name
            }
        }
        
        # Save the results
        self._save_results(results)
        
        logger.info("Experiment completed")
        return results
    
    def _save_results(self, results: Dict):
        """
        Save the experiment results to a JSON file.
        
        Args:
            results: The experiment results
        """
        try:
            results_file = self.output_dir / "results_summary.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM reasoning enhancement experiments")
    parser.add_argument("--dataset", type=str, default="/app/data/rstar/aime2024_test.json", help="Path to the dataset file")
    parser.add_argument("--output-dir", type=str, default="/app/sandbox/writable/run_2025-05-01T18-36-30/results", help="Directory to write results")
    parser.add_argument("--model", type=str, default="qwen3:1.7b", help="Name of the Ollama model to use")
    parser.add_argument("--max-problems", type=int, default=5, help="Maximum number of problems to process per technique")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.dataset, args.output_dir, args.model)
    runner.run_experiment(args.max_problems)