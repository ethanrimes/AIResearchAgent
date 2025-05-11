#!/usr/bin/env python3
"""
Math problem agent that uses the Ollama client to generate solutions.

This script assumes the OllamaClient._make_request method has been updated to handle
line-delimited JSON responses using the following implementation:

def _make_request(self, endpoint: str, data: Dict[str, Any], *, stream: bool = False, timeout: Optional[float] = None) -> Union[requests.Response, Dict[str, Any]]:
    url = self._get_url(endpoint)

    if stream:
        return self._session.post(url, json=data, stream=True, timeout=timeout)
    
    resp = self._session.post(url, json=data, timeout=timeout)
    resp.raise_for_status()
    
    if not resp.content:
        return resp
    
    try:
        # Try parsing as standard JSON first
        return resp.json()
    except json.JSONDecodeError:
        # If that fails, try parsing as line-delimited JSON
        lines = resp.text.strip().split("\n")
        if not lines:
            return {"response": ""}
        
        # Return the last line as JSON (contains the full response)
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            # If all else fails, just return the raw text as the response
            return {"response": resp.text}
"""

import os
import sys
import logging
import time
import requests
from typing import Dict, Any, List, Optional

# Configure logging with both console and file output
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration with detailed formatting"""
    logger = logging.getLogger("math_problem_agent")
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler("math_problem_agent.log")
    file_handler.setLevel(log_level)
    
    # Create detailed formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def set_environment_variables(logger):
    """Set required environment variables based on the configuration"""
    # Convert relative paths to absolute paths
    current_dir = os.getcwd()
    dataset_path = os.path.abspath(os.path.join(current_dir, "data/rstar/aime2024_test.json"))
    directives_path = os.path.abspath(os.path.join(current_dir, "core/directives/test_time_compute_directives"))
    
    # Set environment variables from the provided configuration
    os.environ["DATASET_PATH"] = dataset_path
    os.environ["OLLAMA_DEFAULT_MODEL"] = "o4-mini"
    # Using localhost since we're on the host machine
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["REASONER_MODEL"] = "qwen3:1.7b"
    os.environ["DIRECTIVES_PATH"] = directives_path
    os.environ["MAX_CODING_ITERS"] = "20"
    os.environ["MAX_RUN_DURATION"] = "6000"
    
    # Log all environment variables for debugging
    logger.info("Environment variables set:")
    logger.info(f"DATASET_PATH: {os.environ['DATASET_PATH']}")
    logger.info(f"OLLAMA_DEFAULT_MODEL: {os.environ['OLLAMA_DEFAULT_MODEL']}")
    logger.info(f"OLLAMA_BASE_URL: {os.environ['OLLAMA_BASE_URL']}")
    logger.info(f"REASONER_MODEL: {os.environ['REASONER_MODEL']}")
    logger.info(f"DIRECTIVES_PATH: {os.environ['DIRECTIVES_PATH']}")
    logger.info(f"MAX_CODING_ITERS: {os.environ['MAX_CODING_ITERS']}")
    logger.info(f"MAX_RUN_DURATION: {os.environ['MAX_RUN_DURATION']}")

def check_ollama_models() -> List[str]:
    """Check available Ollama models and return their names"""
    try:
        response = requests.get(f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model.get('name') for model in data.get('models', [])]
        return []
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def format_problem(problem: Dict[str, Any]) -> str:
    """Format a math problem for presentation to the model"""
    return f"Problem ID: {problem['id']}\n\nQuestion: {problem['question']}\n\nPlease solve this math problem step by step."

def main():
    """Main function to load data and generate responses to the first problem"""
    # Initialize logging
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Starting math problem agent")
    logger.info("="*80)
    
    # Set environment variables
    set_environment_variables(logger)
    
    # Check available Ollama models
    available_models = check_ollama_models()
    logger.info(f"Available Ollama models: {available_models}")
    
    # If the configured model isn't available, try to suggest alternatives
    configured_model = os.environ["OLLAMA_DEFAULT_MODEL"]
    if configured_model not in available_models:
        logger.warning(f"Configured model '{configured_model}' is not in the available models list!")
        logger.warning("You need to pull the model first or use an available one")
        
        if available_models:
            logger.info(f"Available models you could use: {', '.join(available_models)}")
            
            # Automatically use the first available model if there are any
            if len(available_models) > 0:
                logger.info(f"Automatically selecting first available model: {available_models[0]}")
                os.environ["OLLAMA_DEFAULT_MODEL"] = available_models[0]
        else:
            logger.error("No models available. Please run 'ollama pull MODEL_NAME' to get a model")
            logger.info("For example: ollama pull mistral or ollama pull llama2")
            return
    
    # Add the module paths to the Python path
    dataloader_path = os.path.abspath("sandbox/read_only/dataloader")
    models_path = os.path.abspath("sandbox/read_only/models")
    
    logger.info(f"Adding to Python path: {dataloader_path}")
    logger.info(f"Adding to Python path: {models_path}")
    
    sys.path.append(dataloader_path)
    sys.path.append(models_path)
    
    try:
        # Import modules after setting Python path
        logger.info("Importing modules from specified paths")
        from rstar_data_loader import rStarDataLoader
        from ollama_client import OllamaClient, GenerationParameters
        
        # Initialize data loader
        logger.info("Initializing rStarDataLoader")
        data_loader = rStarDataLoader()
        
        # Load the dataset
        logger.info(f"Loading dataset from {os.environ['DATASET_PATH']}")
        problems = data_loader.load()
        
        if not problems:
            logger.error("No problems loaded from the dataset!")
            return
        
        logger.info(f"Successfully loaded {len(problems)} problems from dataset")
        
        # Get the first problem
        first_problem = problems[0]
        logger.info(f"Selected first problem: ID={first_problem['id']}")
        logger.info(f"Problem question: {first_problem['question']}")
        
        # Initialize Ollama client
        logger.info("Initializing OllamaClient")
        try:
            ollama_client = OllamaClient()
            logger.info(f"Successfully connected to Ollama server at {ollama_client.base_url}")
            logger.info(f"Using model: {ollama_client.default_model}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return
        
        # Prepare the prompt for the model
        formatted_problem = format_problem(first_problem)
        logger.info(f"Formatted problem for the model:\n{'-'*40}\n{formatted_problem}\n{'-'*40}")
        
        # Generate system prompt
        system_prompt = "You are a mathematics problem-solving agent. Analyze the problem carefully, then provide a clear step-by-step solution. Show all your work and explain your reasoning at each step."
        logger.info(f"Using system prompt: {system_prompt}")
        
        # Create generation parameters
        logger.info("Creating generation parameters")
        gen_params = GenerationParameters(
            model=ollama_client.default_model,  # This will be overridden by the client
            prompt=formatted_problem,
            system=system_prompt,
            temperature=0.2,  # Lower temperature for more deterministic responses
            top_p=0.9,
            num_predict=2048,  # Reasonable limit for the response length
            stream=False,      # Use non-streaming mode for simplicity
        )
        
        # Generate the response
        logger.info("Generating response from Ollama model...")
        logger.info(f"Generating params: {gen_params}")
        start_time = time.time()
        
        try:
            response = ollama_client.generate(gen_params)
            logger.info(f"Response RAW: {response}")
            
            # Log metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Response generated in {elapsed_time:.2f} seconds")
            logger.info(f"Inference metrics: {response.get('metrics', {})}")
            
            # Log the generated response
            generated_text = response.get("response", "")
            logger.info(f"Generated solution:\n{'-'*80}\n{generated_text}\n{'-'*80}")
            
            # Compare with the expected answer if available
            if first_problem.get("answer"):
                logger.info(f"Expected answer: {first_problem['answer']}")
                
            # Print completion message
            print(f"\nSuccessfully generated solution for problem {first_problem['id']}.")
            print(f"Total processing time: {elapsed_time:.2f} seconds")
            print(f"See math_problem_agent.log for detailed logs.")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return
        
        logger.info("="*80)
        logger.info("Process completed successfully")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()