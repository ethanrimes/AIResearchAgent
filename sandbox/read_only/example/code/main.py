"""
Main entry point for running the math reasoning experiment.
"""

import argparse
import logging
from pathlib import Path

from experiment_runner import ExperimentRunner

def main():
    """Run the math reasoning experiment."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Run math reasoning experiment")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results", 
        help="Directory to write results"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3:1.7b", 
        help="Name of the Ollama model to use"
    )
    parser.add_argument(
        "--max_problems", 
        type=int, 
        default=5, 
        help="Maximum number of problems to process per technique"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/experiment.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the experiment
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        model_name=args.model,
        max_problems=args.max_problems
    )
    
    results = runner.run_experiment()
    
    # Print a summary of the results
    best_method = results["experiment_results"]["comparison"]["best_performing"]
    improvement = results["experiment_results"]["comparison"]["accuracy_improvements"].get("H1", 0)
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(f"Baseline accuracy: {results['experiment_results']['baseline']['accuracy']:.2f}")
    print(f"Structured approach accuracy: {results['experiment_results']['hypotheses'][0]['accuracy']:.2f}")
    print(f"Best performing method: {best_method}")
    print(f"Improvement over baseline: {improvement:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()