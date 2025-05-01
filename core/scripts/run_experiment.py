"""
Main entry point for the AI Research Agent.

This script initializes and runs the agent using the configuration provided.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the Agent class
from core.agent.agent import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/app/logs/experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the AI Research Agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="/app/core/config/default_config.yaml", 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--run-dir", 
        type=str, 
        default=None, 
        help="Directory for the current run (default: auto-generated)"
    )
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    args = parse_args()
    
    try:
        # Log startup information
        logger.info("Starting AI Research Agent")
        logger.info(f"Using configuration file: {args.config}")
        
        # Check if run directory was specified from environment variable
        run_dir_from_env = os.environ.get("EXPERIMENT_RUN_PATH")
        if run_dir_from_env:
            logger.info(f"Using run directory from environment: {run_dir_from_env}")
            
        # Ensure environment variables for API access are set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable is not set")
            
        if not os.environ.get("OLLAMA_BASE_URL"):
            logger.warning("OLLAMA_BASE_URL environment variable is not set")
            os.environ["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"
            logger.info(f"Set default OLLAMA_BASE_URL: {os.environ['OLLAMA_BASE_URL']}")
        
        # Override config with environment variables if specified
        config_path = args.config
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
            
        # Initialize the agent with the configuration
        agent = Agent(config_path)
        
        # Run the agent
        results = agent.run()
        
        # Log the results
        if results["status"] == "success":
            logger.info(f"Agent run completed successfully in {results['elapsed_time']:.2f} seconds")
            logger.info(f"Results available in: {agent.workspace['findings']}")
        elif results["status"] == "timeout":
            logger.warning(f"Agent run timed out after step: {results['step']}")
        else:
            logger.error(f"Agent run failed with error: {results['error']}")
            
        # Write the final results to a JSON file
        run_summary_file = Path(agent.workspace["writable"]) / "run_summary.json"
        with open(run_summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Run summary written to: {run_summary_file}")
        
        # Return success
        return 0
        
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())