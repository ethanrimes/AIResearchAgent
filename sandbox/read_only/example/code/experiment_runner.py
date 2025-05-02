"""
Main experiment runner for testing LLM math reasoning enhancement techniques.

This script loads math problems from a dataset, tests different reasoning techniques,
and evaluates the results according to the specified schema.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the required modules from read-only environment
from sandbox.read_only.dataloader.rstar_data_loader import rStarDataLoader
from sandbox.read_only.evaluator.answer_evaluation import AnswerEvaluator

# Import the solvers
from baseline import BaselineSolver
from structured_approach import StructuredSolver

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
    
    def __init__(self, output_dir: str, max_problems: int = 5):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to write results
            max_problems: Maximum number of problems to process per technique
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_problems = max_problems
        
        # Initialize components
        self.data_loader = rStarDataLoader()
        self.evaluator = AnswerEvaluator()
        self.baseline_solver = BaselineSolver()
        self.structured_solver = StructuredSolver()
        
        logger.info("ExperimentRunner initialized")
        logger.info(f"Will process up to {max_problems} problems per technique")
    
    def run_experiment(self) -> Dict:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary with experiment results in the required schema
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
        
        # Run structured approach (H1)
        structured_results = self.run_hypothesis_h1(test_problems)
        
        # Compute comparison metrics
        comparison = self.compute_comparison(baseline_results, [structured_results])
        
        # Compile the overall results according to the required schema
        results = {
            "experiment_results": {
                "baseline": baseline_results,
                "hypotheses": [structured_results],
                "comparison": comparison
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
        
        test_questions = []
        execution_times = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Processing baseline problem {i+1}/{len(problems)}: {problem['id']}")
            
            # Process the problem
            model_output, elapsed_time = self.baseline_solver.solve(problem)
            execution_times.append(elapsed_time)
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate(
                model_output, problem['answer']
            )
            
            # Record the result
            test_questions.append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Calculate summary statistics
        total_problems = len(test_questions)
        correct_answers = sum(1 for q in test_questions if q.get("is_correct", False))
        accuracy = round(correct_answers / total_problems, 2) if total_problems > 0 else 0.0
        average_time = round(sum(execution_times) / len(execution_times), 2) if execution_times else 0.0
        
        results = {
            "accuracy": accuracy,
            "test_questions": test_questions,
            "summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time
            }
        }
        
        logger.info(f"Baseline completed with accuracy: {accuracy}")
        return results
    
    def run_hypothesis_h1(self, problems: List[Dict]) -> Dict:
        """
        Run the structured approach (H1).
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            Dictionary with results in the required format
        """
        logger.info("Running structured approach (H1)")
        
        test_questions = []
        execution_times = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Processing H1 problem {i+1}/{len(problems)}: {problem['id']}")
            
            # Process the problem
            model_output, elapsed_time = self.structured_solver.solve(problem)
            execution_times.append(elapsed_time)
            
            # Evaluate the response
            extracted_answer, is_correct = self.evaluator.evaluate(
                model_output, problem['answer']
            )
            
            # Record the result
            test_questions.append({
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected_answer": problem['answer'],
                "model_answer": extracted_answer if extracted_answer else "No answer extracted",
                "is_correct": is_correct,
                "execution_time": round(elapsed_time, 2)
            })
        
        # Calculate summary statistics
        total_problems = len(test_questions)
        correct_answers = sum(1 for q in test_questions if q.get("is_correct", False))
        accuracy = round(correct_answers / total_problems, 2) if total_problems > 0 else 0.0
        average_time = round(sum(execution_times) / len(execution_times), 2) if execution_times else 0.0
        
        results = {
            "id": "H1",
            "accuracy": accuracy,
            "test_questions": test_questions,
            "summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time
            }
        }
        
        logger.info(f"H1 completed with accuracy: {accuracy}")
        return results
    
    def compute_comparison(self, baseline: Dict, hypotheses: List[Dict]) -> Dict:
        """
        Compare hypotheses against the baseline.
        
        Args:
            baseline: Baseline results
            hypotheses: List of hypothesis results
            
        Returns:
            Comparison metrics
        """
        baseline_accuracy = baseline["accuracy"]
        
        # Compute accuracy improvements
        accuracy_improvements = {}
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis["id"]
            improvement = round(hypothesis["accuracy"] - baseline_accuracy, 2)
            accuracy_improvements[hypothesis_id] = improvement
        
        # Find the best performing method
        all_methods = [{"id": "Baseline", "accuracy": baseline_accuracy}] + [
            {"id": h["id"], "accuracy": h["accuracy"]} for h in hypotheses
        ]
        best_method = max(all_methods, key=lambda x: x["accuracy"])["id"]
        
        return {
            "accuracy_improvements": accuracy_improvements,
            "best_performing": best_method
        }
    
    def save_results(self, results: Dict):
        """
        Save the experiment results to a JSON file.
        
        Args:
            results: The experiment results
        """
        results_file = self.output_dir / "results_summary.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

# Main function to run the experiment directly
def main():
    """Main function for running the experiment when executed directly."""
    # Hardcoded values as per requirements
    output_dir = "run_2025-05-01T18-36-30/results"
    max_problems = 5  # Simple default value
    
    # Run the experiment
    runner = ExperimentRunner(
        output_dir=output_dir,
        max_problems=max_problems
    )
    
    results = runner.run_experiment()
    
    # Print summary
    if "experiment_results" in results:
        baseline_acc = results["experiment_results"]["baseline"]["accuracy"]
        h1_acc = results["experiment_results"]["hypotheses"][0]["accuracy"]
        best = results["experiment_results"]["comparison"]["best_performing"]
        
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        print(f"Baseline accuracy: {baseline_acc:.2f}")
        print(f"H1 accuracy: {h1_acc:.2f}")
        print(f"Best performing method: {best}")
        print("="*50)

if __name__ == "__main__":
    main()