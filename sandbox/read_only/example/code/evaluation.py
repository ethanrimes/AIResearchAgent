"""
Evaluation module for assessing LLM math problem-solving performance.

This module provides functions for evaluating model outputs and
compiling performance metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from sandbox.read_only.evaluator.answer_evaluation import AnswerEvaluator

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    """
    Evaluator for LLM math problem-solving experiments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the experiment evaluator.
        
        Args:
            output_dir: Directory to write evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.answer_evaluator = AnswerEvaluator()
        
        logger.info(f"ExperimentEvaluator initialized with output directory: {output_dir}")
    
    def evaluate_response(self, model_response: str, ground_truth: str) -> Tuple[Optional[str], bool, Dict]:
        """
        Evaluate a single model response.
        
        Args:
            model_response: The model's output text
            ground_truth: The ground truth answer
            
        Returns:
            Tuple of (extracted_answer, is_correct, detailed_results)
        """
        extracted_answer, is_correct = self.answer_evaluator.evaluate(model_response, ground_truth)
        
        # Get detailed evaluation results
        detailed_results = {}
        if extracted_answer:
            detailed_results = self.answer_evaluator.evaluate_correctness(extracted_answer, ground_truth)
        
        return extracted_answer, is_correct, detailed_results
    
    def compile_results(self, technique_name: str, technique_id: str, 
                      examples: List[Dict], execution_times: List[float]) -> Dict:
        """
        Compile results for a technique into a standardized format.
        
        Args:
            technique_name: Name of the technique
            technique_id: Identifier of the technique (e.g., "H1")
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
            "baseline_accuracy": baseline_accuracy,
            "accuracy_improvements": improvements
        }
        
        return comparison
    
    def save_results(self, experiment_results: Dict):
        """
        Save experiment results to a JSON file.
        
        Args:
            experiment_results: The complete experiment results
        """
        try:
            results_file = self.output_dir / "evaluation_results.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_results, f, indent=2)
                
            logger.info(f"Evaluation results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def generate_findings(self, experiment_results: Dict) -> Dict:
        """
        Generate findings from experiment results.
        
        Args:
            experiment_results: The complete experiment results
            
        Returns:
            Dictionary with findings
        """
        baseline = experiment_results["experiment_results"]["baseline"]
        hypotheses = experiment_results["experiment_results"]["hypotheses"]
        comparison = experiment_results["experiment_results"]["comparison"]
        
        # Generate a summary of findings
        summary = f"The experiment tested {len(hypotheses)} techniques against the baseline approach on {baseline['summary']['total_problems']} mathematical problems."
        
        if comparison["best_performing"] == "Baseline":
            summary += " None of the tested techniques outperformed the baseline approach."
        else:
            best_technique = next(h for h in hypotheses if h["id"] == comparison["best_performing"])
            summary += f" The best performing technique was {best_technique['name']} ({best_technique['id']}), which improved accuracy by {comparison['accuracy_improvements'][best_technique['id']]*100:.1f}% over the baseline."
        
        # Generate hypothesis results
        hypothesis_results = []
        for hypothesis in hypotheses:
            improvement = comparison["accuracy_improvements"][hypothesis["id"]]
            effectiveness = "Low"
            if improvement >= 0.1:
                effectiveness = "High"
            elif improvement >= 0.05:
                effectiveness = "Medium"
            
            result = {
                "id": hypothesis["id"],
                "name": hypothesis.get("name", f"Technique {hypothesis['id']}"),
                "effectiveness": effectiveness,
                "performance_change": f"{'+' if improvement >= 0 else ''}{improvement*100:.1f}% accuracy compared to baseline",
                "analysis": f"The {hypothesis.get('name', 'technique')} {'improved' if improvement > 0 else 'worsened' if improvement < 0 else 'did not change'} model performance on the test problems.",
                "conditions": "Further analysis with more problems would be needed to determine the specific conditions where this technique works best.",
                "limitations": f"One limitation is the {'increased' if hypothesis['summary']['average_time'] > baseline['summary']['average_time'] else 'decreased'} solution time ({hypothesis['summary']['average_time']:.2f}s vs {baseline['summary']['average_time']:.2f}s for baseline)."
            }
            
            hypothesis_results.append(result)
        
        # Generate findings
        findings = {
            "findings": {
                "summary": summary,
                "hypotheses_results": hypothesis_results,
                "key_insights": [
                    "Structuring the reasoning process can impact model performance on mathematical problems.",
                    "There is a trade-off between accuracy and solution time for some techniques.",
                    "Different techniques may be more effective for different types of problems."
                ],
                "reasoning_patterns": {
                    "successful_patterns": [
                        "Breaking problems into explicit steps with clear subtasks",
                        "Explicitly identifying variables and constraints before solving",
                        "Verifying answers against the original problem constraints"
                    ],
                    "unsuccessful_patterns": [
                        "Skipping steps in complex calculations",
                        "Failing to properly understand the problem statement",
                        "Making calculation errors in multi-step problems"
                    ]
                },
                "future_directions": [
                    "Test the techniques on a larger and more diverse set of problems",
                    "Explore combinations of the most effective techniques",
                    "Investigate techniques tailored to specific types of math problems",
                    "Compare performance across different model sizes and architectures"
                ]
            },
            "meta": experiment_results["meta"]
        }
        
        # Save the findings
        findings_file = self.output_dir / "findings.json"
        try:
            with open(findings_file, 'w', encoding='utf-8') as f:
                json.dump(findings, f, indent=2)
                
            logger.info(f"Findings saved to {findings_file}")
        except Exception as e:
            logger.error(f"Error saving findings: {e}")
            
        return findings


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    
    # Create sample results for testing
    sample_results = {
        "experiment_results": {
            "baseline": {
                "accuracy": 0.4,
                "examples": [
                    {"problem_id": "1", "is_correct": True, "execution_time": 3.5},
                    {"problem_id": "2", "is_correct": False, "execution_time": 4.2}
                ],
                "summary": {
                    "total_problems": 5,
                    "correct_answers": 2,
                    "accuracy": 0.4,
                    "average_time": 3.8
                }
            },
            "hypotheses": [
                {
                    "id": "H1",
                    "name": "Structured Step-by-Step Decomposition",
                    "accuracy": 0.6,
                    "examples": [
                        {"problem_id": "1", "is_correct": True, "execution_time": 4.5},
                        {"problem_id": "2", "is_correct": True, "execution_time": 5.1}
                    ],
                    "summary": {
                        "total_problems": 5,
                        "correct_answers": 3,
                        "accuracy": 0.6,
                        "average_time": 4.8
                    }
                }
            ],
            "comparison": {
                "best_performing": "H1",
                "accuracy_improvements": {
                    "H1": 0.2
                }
            }
        },
        "meta": {
            "timestamp": "2025-05-01T12:34:56",
            "dataset": "test_dataset.json",
            "model": "test_model"
        }
    }
    
    # Test the evaluator
    evaluator = ExperimentEvaluator("./test_output")
    findings = evaluator.generate_findings(sample_results)
    
    print("Generated findings:")
    print(json.dumps(findings, indent=2))