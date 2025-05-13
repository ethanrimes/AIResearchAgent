"""
Validation checklist descriptions for the experiment implementation.

This module provides descriptions and utility functions for validation requirements
used in evaluating experiment code implementations.
"""

from typing import Dict, Any

# Descriptions for each requirement in the validation checklist
REQUIREMENT_DESCRIPTIONS = {
    "hypotheses_implementation": "For each hypothesis in the original experiment plan, this assesses whether it is fully implemented in the experiment code",
    "baseline_calculation": "Whether the baseline case (no improvement on inference ability) is implemented in the experiment code",
    "correct_entrypoint": "Whether /experiment/run_experiment.sh is the correct entrypoint for this experiment. Running this bash script should run the whole experiment",
    "final_results_metrics": "Whether the experiment has produced a results/results_summary.json object in accordance with the expected schema",
    "final_results_authenticity": "Whether these results are truly the output of the experiment rather than hard-coded or hallucinated values",
    "dataloader_api_usage": "Whether the experiment successfully imports and uses 'from dataloader.rstar_data_loader import rStarDataLoader' to load the test data",
    "ollama_client_api_usage": "Whether the experiment successfully imports and uses 'from models.ollama_client import OllamaClient, GenerationParameters' to run inference",
    "extractor_evaluator_api_usage": "Whether the experiment successfully imports and uses 'from evaluator.answer_evaluation import AnswerEvaluator' to extract answers from generations and evaluate their correctness",
    "sufficient_answer_length": "Whether the generations have enough tokens to generate complete answers or if the '\\boxed{{}}' answer statements are cut off",
    "no_cheating": "Whether the algorithm avoids exposing the ground truth to the model unfairly or using ground truth unfairly in answer calculations"
}

# Descriptions for manual validation checks
MANUAL_VALIDATION_DESCRIPTIONS = {
    "entry_script_exists": "Whether the entry script /experiment/run_experiment.sh exists and is not empty",
    "python_files_exist": "Whether Python files exist in the experiment directory",
    "results_file_exists": "Whether the results_summary.json file exists in the results directory"
}

def get_requirement_description(requirement_key: str) -> str:
    """
    Get the description for a requirement key.
    
    Args:
        requirement_key: Key of the requirement.
        
    Returns:
        Description of the requirement.
    """
    return REQUIREMENT_DESCRIPTIONS.get(requirement_key, "No description available")

def get_manual_validation_description(validation_key: str) -> str:
    """
    Get the description for a manual validation check.
    
    Args:
        validation_key: Key of the manual validation check.
        
    Returns:
        Description of the manual validation check.
    """
    return MANUAL_VALIDATION_DESCRIPTIONS.get(validation_key, "No description available")