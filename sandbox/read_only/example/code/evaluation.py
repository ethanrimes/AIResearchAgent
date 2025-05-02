"""
Evaluation module for the math reasoning experiment.

This module uses the AnswerEvaluator from the read-only environment
to evaluate model outputs and format results according to the required schema.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import the answer evaluator from the read-only environment
from sandbox.read_only.evaluator.answer_evaluation import AnswerEvaluator

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    """
    Evaluator for the math reasoning experiment results.
    Formats data according to the required schema.
    """
    
    def __init__(self):
        """Initialize the experiment evaluator with an answer evaluator."""
        self.answer_evaluator = AnswerEvaluator()
        logger.info("ExperimentEvaluator initialized")
    
    def evaluate_response(self, model_response: str, ground_truth: str) -> Tuple[Optional[str], bool]:
        """
        Evaluate a single model response against the ground truth.
        
        Args:
            model_response: The model's output text
            ground_truth: The ground truth answer
            
        Returns:
            Tuple of (extracted_answer, is_correct)
        """
        return self.answer_evaluator.evaluate(model_response, ground_truth)
    
    def format_results(self, baseline_data: Dict[str, Any], hypotheses_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the complete experiment results according to the required schema.
        
        Args:
            baseline_data: Dictionary with baseline data
            hypotheses_data: List of dictionaries with hypotheses data
            
        Returns:
            Formatted results dictionary matching the required schema
        """
        # Compute comparison metrics
        comparison = self.compare_techniques(baseline_data, hypotheses_data)
        
        # Format the final output according to the exact schema
        return {
            "experiment_results": {
                "baseline": baseline_data,
                "hypotheses": hypotheses_data
            }
        }
    
    def process_baseline_results(self, test_questions: List[Dict], 
                               execution_times: List[float]) -> Dict[str, Any]:
        """
        Process baseline results according to the required schema.
        
        Args:
            test_questions: List of test question results
            execution_times: List of execution times
            
        Returns:
            Dictionary with formatted baseline results
        """
        # Calculate summary statistics
        total_problems = len(test_questions)
        correct_answers = sum(1 for q in test_questions if q.get("is_correct", False))
        accuracy = round(correct_answers / total_problems, 2) if total_problems > 0 else 0.0
        average_time = round(sum(execution_times) / len(execution_times), 2) if execution_times else 0.0
        
        return {
            "accuracy": accuracy,
            "test_questions": test_questions,
            "summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time
            }
        }
    
    def process_hypothesis_results(self, hypothesis_id: str, test_questions: List[Dict], 
                                 execution_times: List[float]) -> Dict[str, Any]:
        """
        Process hypothesis results according to the required schema.
        
        Args:
            hypothesis_id: Identifier for the hypothesis (e.g., "H1")
            test_questions: List of test question results
            execution_times: List of execution times
            
        Returns:
            Dictionary with formatted hypothesis results
        """
        # Calculate summary statistics
        total_problems = len(test_questions)
        correct_answers = sum(1 for q in test_questions if q.get("is_correct", False))
        accuracy = round(correct_answers / total_problems, 2) if total_problems > 0 else 0.0
        average_time = round(sum(execution_times) / len(execution_times), 2) if execution_times else 0.0
        
        return {
            "id": hypothesis_id,
            "accuracy": accuracy,
            "test_questions": test_questions,
            "summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time
            }
        }
    
    def compare_techniques(self, baseline_results: Dict[str, Any], 
                          technique_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance of different techniques against the baseline.
        
        Args:
            baseline_results: Results from the baseline technique
            technique_results: Results from other techniques
            
        Returns:
            Dictionary with comparison results
        """
        # Find the best performing technique
        all_results = [baseline_results] + technique_results
        best_technique = max(all_results, key=lambda x: x["accuracy"])
        
        # For baseline, use "Baseline" as the ID
        best_id = "Baseline" if best_technique is baseline_results else best_technique["id"]
        
        # Calculate improvements over baseline
        baseline_accuracy = baseline_results["accuracy"]
        improvements = {}
        
        for technique in technique_results:
            technique_id = technique["id"]
            improvement = round(technique["accuracy"] - baseline_accuracy, 2)
            improvements[technique_id] = improvement
        
        # Compile comparison results
        comparison = {
            "best_performing": best_id,
            "accuracy_improvements": improvements
        }
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save the experiment results to a JSON file.
        
        Args:
            results: The experiment results
            output_file: Path to the output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")