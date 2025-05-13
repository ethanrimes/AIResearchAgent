"""
Main entry point for the AI Research Agent.

This script initializes and runs the agent using the configuration provided.
"""
import pytest
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
        from pathlib import Path
        log_path = Path("/logs")
        logger = setup_logging(run_dir=log_path)
        
        # Log startup information
        logger.info("Starting AI Research Agent")
        logger.info(f"Using configuration file: {args.config}")
        logger.info("Using flattened directory structure at root level")
        
        # Load configuration from file
        config_path = args.config
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Read the config file directly (Agent will use the centralized configuration)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set timeout from config if available (for backwards compatibility)
        if "max_run_duration" in config:
            logger.info(f"Using timeout from config: {config['max_run_duration']}s")
            os.environ["EXPERIMENT_TIMEOUT"] = str(config["max_run_duration"])
        
        # Ensure environment variables for API access are set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable is not set")
            
        if not os.environ.get("OLLAMA_BASE_URL"):
            logger.warning("OLLAMA_BASE_URL environment variable is not set")
            os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
            logger.info(f"Set default OLLAMA_BASE_URL: {os.environ['OLLAMA_BASE_URL']}")
        
        # Update workspace paths to use flattened directory structure in the config
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

        # Run additional component tests after agent initialization
        logger.info("Running additional component tests")
        try:
            # Add the integration_tests directory to the path
            import sys
            from pathlib import Path
            integration_tests_dir = Path("/app/infra/docker/integration_tests")
            sys.path.append(str(integration_tests_dir))
            
            # Now import the module directly
            from run_integration_tests import run_additional_tests
            run_additional_tests(agent)
        except ImportError as e:
            logger.warning(f"Could not import run_additional_tests module: {e}, skipping additional tests")

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
        run_summary_file = Path("/logs/agent_summary_metrics.json")
        with open(run_summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Run summary written to: {run_summary_file}")
        
        # Return success
        return 0
        
    except Exception as e:
        from pathlib import Path
        log_path = Path("/logs")
        # If an exception occurs before logging is set up, set it up now
        if not logging.getLogger().handlers:
            logger = setup_logging(run_dir=log_path)
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())