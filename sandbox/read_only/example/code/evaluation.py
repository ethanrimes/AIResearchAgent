"""
Evaluation module for the experiment.

This module provides functions for evaluating model outputs and
compiling performance metrics.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

from sandbox.read_only.evaluator.answer_evaluation import AnswerEvaluator

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    """
    Evaluator for the math reasoning experiment.
    """
    
    def __init__(self):
        """Initialize the experiment evaluator."""
        self.answer_evaluator = AnswerEvaluator()
        logger.info("ExperimentEvaluator initialized")
    
    def evaluate_response(self, model_response: str, ground_truth: str) -> Tuple[Optional[str], bool]:
        """
        Evaluate a single model response.
        
        Args:
            model_response: The model's output text
            ground_truth: The ground truth answer
            
        Returns:
            Tuple of (extracted_answer, is_correct)
        """
        return self.answer_evaluator.evaluate(model_response, ground_truth)
    
    def compile_results(self, technique_name: str, technique_id: str, 
                      examples: List[Dict], execution_times: List[float]) -> Dict:
        """
        Compile results for a technique into a standardized format.
        
        Args:
            technique_name: Name of the technique
            technique_id: Identifier of the technique (e.g., "H1" or "Baseline")
            examples: List of problem examples with results
            execution_times: List of execution times
            
        Returns:
            Dictionary with compiled results
        """
        total_problems = len(examples)
        correct_answers = sum(1 for ex in examples if ex.get("is_correct", False))
        
        accuracy = 0.0
        average_time = 0.0
        
        if total_problems > 0:
            accuracy = round(correct_answers / total_problems, 2)
            average_time = round(sum(execution_times) / len(execution_times), 2) if execution_times else 0.0
        
        results = {
            "id": technique_id,
            "name": technique_name,
            "accuracy": accuracy,
            "examples": examples,
            "summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time
            }
        }
        
        return results
    
    def compare_techniques(self, baseline_results: Dict, technique_results: List[Dict]) -> Dict:
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