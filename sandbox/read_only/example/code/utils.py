"""
Utility functions for the math reasoning experiment.

This module provides helper functions for loading data, formatting outputs,
and visualizing results.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the JSON contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}
        
def save_json(data: Dict, file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def create_results_visualizations(results: Dict, output_dir: str):
    """
    Create visualizations of experiment results.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save visualizations
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract relevant data
        baseline_accuracy = results["experiment_results"]["baseline"]["accuracy"]
        hypotheses = results["experiment_results"]["hypotheses"]
        
        technique_names = ["Baseline"] + [h["id"] for h in hypotheses]
        accuracies = [baseline_accuracy] + [h["accuracy"] for h in hypotheses]
        
        # Create accuracy comparison bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(technique_names, accuracies, color=['gray'] + ['blue'] * len(hypotheses))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, max(accuracies) + 0.1)
        plt.title('Accuracy Comparison Across Techniques')
        plt.ylabel('Accuracy')
        plt.xlabel('Technique')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the chart
        accuracy_chart_path = output_path / "accuracy_comparison.png"
        plt.savefig(accuracy_chart_path)
        plt.close()
        
        logger.info(f"Saved accuracy comparison chart to {accuracy_chart_path}")
        
        # Create execution time comparison
        baseline_time = results["experiment_results"]["baseline"]["summary"]["average_time"]
        execution_times = [baseline_time] + [h["summary"]["average_time"] for h in hypotheses]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(technique_names, execution_times, color=['gray'] + ['green'] * len(hypotheses))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.ylim(0, max(execution_times) + 1)
        plt.title('Average Execution Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Technique')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the chart
        time_chart_path = output_path / "time_comparison.png"
        plt.savefig(time_chart_path)
        plt.close()
        
        logger.info(f"Saved execution time comparison chart to {time_chart_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def format_experiment_summary(results: Dict) -> str:
    """
    Format experiment results into a readable summary.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Formatted summary string
    """
    try:
        baseline = results["experiment_results"]["baseline"]
        hypotheses = results["experiment_results"]["hypotheses"]
        comparison = results["experiment_results"]["comparison"]
        meta = results["meta"]
        
        # Format header
        summary = f"# Math Reasoning Enhancement Experiment Summary\n\n"
        summary += f"**Date:** {meta['timestamp']}\n"
        summary += f"**Dataset:** {meta['dataset']}\n"
        summary += f"**Model:** {meta['model']}\n\n"
        
        # Format baseline results
        summary += f"## Baseline Results\n\n"
        summary += f"- **Accuracy:** {baseline['accuracy']:.2f}\n"
        summary += f"- **Problems Solved:** {baseline['summary']['correct_answers']}/{baseline['summary']['total_problems']}\n"
        summary += f"- **Average Execution Time:** {baseline['summary']['average_time']:.2f}s\n\n"
        
        # Format hypotheses results
        summary += f"## Technique Results\n\n"
        
        for h in hypotheses:
            improvement = comparison["accuracy_improvements"][h["id"]]
            summary += f"### {h.get('name', h['id'])}\n\n"
            summary += f"- **Accuracy:** {h['accuracy']:.2f} ({'+' if improvement > 0 else ''}{improvement:.2f} over baseline)\n"
            summary += f"- **Problems Solved:** {h['summary']['correct_answers']}/{h['summary']['total_problems']}\n"
            summary += f"- **Average Execution Time:** {h['summary']['average_time']:.2f}s\n\n"
        
        # Format comparison
        summary += f"## Overall Comparison\n\n"
        summary += f"- **Best Performing Technique:** {comparison['best_performing']}\n"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error formatting experiment summary: {e}")
        return f"Error generating summary: {str(e)}"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample results
    sample_results = {
        "experiment_results": {
            "baseline": {
                "accuracy": 0.4,
                "examples": [],
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
                    "examples": [],
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
    
    # Test the utilities
    print(format_experiment_summary(sample_results))
    create_results_visualizations(sample_results, "./test_output")