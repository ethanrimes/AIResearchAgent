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
from core.utils.path_utils import get_active_run_timestamp


# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the Agent class
from core.agent.agent import Agent
# Import the logging utility
from core.utils.logging_utils import setup_logging

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
        # Import your path utility
        from core.utils.path_utils import get_active_run_path
        
        # Use the existing run directory instead of creating a new one
        if args.run_dir:
            # If run directory was explicitly specified via args
            run_dir = Path(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Get the run directory that was already created by the entrypoint script
            try:
                run_dir = get_active_run_path()
                # No need to mkdir as it should already exist
            except FileNotFoundError:
                # Fallback only if the file doesn't exist (shouldn't happen if entrypoint.sh ran correctly)
                logger = setup_logging()
                logger.warning("Active run path file not found, creating a new run directory")
                run_dir = Path("/app/sandbox/writable") / f"run_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
                run_dir.mkdir(parents=True, exist_ok=True)
            
        # Set up logging using the utility function
        logger = setup_logging(run_dir=run_dir)
            
        # Log startup information
        logger.info("Starting AI Research Agent")
        logger.info(f"Using configuration file: {args.config}")
        logger.info(f"Using run directory: {run_dir}")
        
        # No need to create active_run_path_file as it should already exist
        # Just log that we're using it
        logger.info(f"Using active run path: {run_dir}")
            
        # Set up logging using the utility function
        logger = setup_logging(run_dir=run_dir)
            
        # Log startup information
        logger.info("Starting AI Research Agent")
        logger.info(f"Using configuration file: {args.config}")
        logger.info(f"Using run directory: {run_dir}")
        
        # Create the active run path file
        active_run_path_file = Path("/app/active_run_path/path.txt")
        active_run_path_file.parent.mkdir(parents=True, exist_ok=True)
        with open(active_run_path_file, 'w') as f:
            f.write(str(run_dir))
        logger.info(f"Set active run path to: {run_dir}")
        
        # Check if run directory was specified from environment variable
        run_dir_from_env = os.environ.get("EXPERIMENT_RUN_PATH")
        if run_dir_from_env:
            logger.info(f"Using run directory from environment: {run_dir_from_env}")
            
        # Load configuration from file
        config_path = args.config
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
            
        # Add this to run_experiment.py after loading config
        if "max_run_duration" in config:
            logger.info(f"Using timeout from config: {config['max_run_duration']}s")
            os.environ["EXPERIMENT_TIMEOUT"] = str(config["max_run_duration"])
        
        # Read the config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override config values with environment variables if they exist
        if os.environ.get("DATASET_PATH"):
            logger.info(f"Overriding dataset_path from environment: {os.environ['DATASET_PATH']}")
            config["dataset_path"] = os.environ["DATASET_PATH"]
        
        if os.environ.get("AGENT_MODEL"):
            logger.info(f"Overriding agent_model from environment: {os.environ['AGENT_MODEL']}")
            config["agent_model"] = os.environ["AGENT_MODEL"]
            
        if os.environ.get("REASONER_MODEL"):
            logger.info(f"Overriding reasoner_model from environment: {os.environ['REASONER_MODEL']}")
            config["reasoner_model"] = os.environ["REASONER_MODEL"]
        
        # Ensure environment variables for API access are set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable is not set")
            
        if not os.environ.get("OLLAMA_BASE_URL"):
            logger.warning("OLLAMA_BASE_URL environment variable is not set")
            os.environ["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"
            logger.info(f"Set default OLLAMA_BASE_URL: {os.environ['OLLAMA_BASE_URL']}")
        
        # Write modified config back to a temporary file
        temp_config_path = Path("/tmp/modified_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Initialize the agent with the modified configuration
        agent = Agent(str(temp_config_path))
        
        # Run the agent
        results = agent.run()
        
        # Rest of the function remains the same
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
        # If an exception occurs before logging is set up, set it up now
        if not logging.getLogger().handlers:
            logger = setup_logging()
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())