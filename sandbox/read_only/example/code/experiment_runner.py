"""
Main experiment runner for testing LLM math reasoning enhancement techniques.

This script loads math problems from a dataset, tests different reasoning techniques,
and evaluates the results.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Import the data loader
from sandbox.read_only.dataloader.rstar_data_loader import rStarDataLoader

# Import the solvers
from baseline import BaselineSolver
from structured_approach import StructuredSolver
from evaluation import ExperimentEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Runner for the math reasoning experiment.
    """
    
    def __init__(self, output_dir: str, model_name: str = "qwen3:1.7b", max_problems: int = 5):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to write results
            model_name: Name of the Ollama model to use
            max_problems: Maximum number of problems to process per technique
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_problems = max_problems
        
        # Initialize components
        self.data_loader = rStarDataLoader()
        self.evaluator = ExperimentEvaluator()
        self.baseline_solver = BaselineSolver(model_name)
        self.structured_solver = StructuredSolver(model_name)
        
        logger.info(f"ExperimentRunner initialized with model: {model_name}")
        logger.info(f"Will process up to {max_problems} problems per technique")
    
    def run_experiment(self) -> Dict:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary with experiment results
        """
        logger.info("Starting math reasoning experiment")
        
        # Load the problems
        problems = self.data_loader.load()
        
        if not problems:
            logger.error("No problems loaded from dataset")
            return {"error": "No problems loaded from dataset"}
        
        logger.info(f"Loaded {len(problems)} problems from dataset")
        
        # Limit the number of problems
        test_problems = problems[:self.max_problems]
        logger.info(f"Using {len(test_problems)} problems for testing")
        
        # Run baseline
        baseline_results = self.run_baseline(test_problems)
        
        # Run structured approach
        structured_results = self.run_structured_approach(test_problems)
        
        # Compare techniques
        comparison = self.evaluator.compare_techniques(
            baseline_results, [structured_results]
        )
        
        # Compile the overall results
        results = {
            "experiment_results": {
                "baseline": baseline_results,
                "hypotheses": [structured_results],
                "comparison": comparison
            },
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset": os.environ.get("DATASET_PATH", "unknown"),
                "model": self.model_name
            }
        }
        
        # Save the results
        self.save_results(results)
        
        logger.info("Experiment completed successfully")
        return results
    
    def run_baseline(self, problems: List[Dict]) -> Dict:
        """
        Run the baseline approach.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            Dictionary with baseline results
        """
        logger.info("Running baseline approach")
        
        examples = []
        execution_times = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Processing baseline problem {i+1}/{len(problems)}: {problem['id']}")
            
            # Process the problem
            model_output, elapsed_time = self.baseline_solver.solve(problem)
            execution_times.append(elapsed_time)
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate_response(
                model_output, problem['answer']
            )
            
            # Record the result
            examples.append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Compile the results
        results = self.evaluator.compile_results(
            technique_name="Standard Approach (Baseline)",
            technique_id="Baseline",
            examples=examples,
            execution_times=execution_times
        )
        
        logger.info(f"Baseline completed with accuracy: {results['accuracy']}")
        return results
    
    def run_structured_approach(self, problems: List[Dict]) -> Dict:
        """
        Run the structured approach.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            Dictionary with results
        """
        logger.info("Running structured approach")
        
        examples = []
        execution_times = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Processing structured approach problem {i+1}/{len(problems)}: {problem['id']}")
            
            # Process the problem
            model_output, elapsed_time = self.structured_solver.solve(problem)
            execution_times.append(elapsed_time)
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate_response(
                model_output, problem['answer']
            )
            
            # Record the result
            examples.append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Compile the results
        results = self.evaluator.compile_results(
            technique_name="Structured Step-by-Step Decomposition",
            technique_id="H1",
            examples=examples,
            execution_times=execution_times
        )
        
        logger.info(f"Structured approach completed with accuracy: {results['accuracy']}")
        return results
    
    def save_results(self, results: Dict):
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