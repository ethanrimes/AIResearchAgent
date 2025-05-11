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
        # Set up logging directly with root paths
        logger = setup_logging(run_dir=Path("/logs"))
        
        # Log startup information
        logger.info("Starting AI Research Agent")
        logger.info(f"Using configuration file: {args.config}")
        logger.info("Using flattened directory structure at root level")
        
        # Load configuration from file
        config_path = args.config
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Read the config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set timeout from config if available
        if "max_run_duration" in config:
            logger.info(f"Using timeout from config: {config['max_run_duration']}s")
            os.environ["EXPERIMENT_TIMEOUT"] = str(config["max_run_duration"])
        
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
        
        # Update workspace paths to use flattened directory structure
        logger.info("Configuring workspace with flattened directory structure")
        config['workspace'] = {
            'experiment': '/experiment',
            'findings': '/findings',
            'hypothesis': '/hypothesis',
            'planning': '/planning',
            'results': '/results',
            'read_only': '/read_only',
            'logs': '/logs',
            'scratchpad': '/scratchpad',
            'writable': '/logs'  # Use logs directory for writable files like run_summary.json
        }
        logger.info("Workspace paths configured for flattened structure")

        # Write modified config back to a temporary file
        temp_config_path = Path("/tmp/modified_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Initialize the agent with the modified configuration
        logger.info("Initializing AI Research Agent")
        agent = Agent(str(temp_config_path))
        
        # Run the agent
        logger.info("Starting agent execution")
        results = agent.run()
        
        # Process results
        if results["status"] == "success":
            logger.info(f"Agent run completed successfully in {results['elapsed_time']:.2f} seconds")
            logger.info(f"Results available in: {agent.workspace['findings']}")
        elif results["status"] == "timeout":
            logger.warning(f"Agent run timed out after step: {results['step']}")
        else:
            logger.error(f"Agent run failed with error: {results['error']}")
            
        # Write the final results to a JSON file in the /results directory
        run_summary_file = Path("/results/run_summary.json")
        with open(run_summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Run summary written to: {run_summary_file}")
        
        # Return success
        return 0
        
    except Exception as e:
        # If an exception occurs before logging is set up, set it up now
        if not logging.getLogger().handlers:
            logger = setup_logging(run_dir=Path("/logs"))
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())