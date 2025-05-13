import json
import logging
from typing import Dict, List, Any
import os
import sys
from pathlib import Path

from core.agent.agent_config import AgentConfig  # Import the centralized config manager

class rStarDataLoader:
    """
    Simple data loader for math problem datasets in JSON format.
    """
    def __init__(self):
        """
        Initialize the data loader.
        """
        # Set up module-specific logger
        self.logger = logging.getLogger("ai_research_agent.rstar_data_loader")
        self._setup_logger()
        # Get configuration from the registry using class method
        # This properly uses the AgentConfig as a singleton with class methods
        default_config_path = "/app/core/config/default_config.yaml"
        if os.path.exists(default_config_path):
            self.logger.info(f"Loading configuration from default path: {default_config_path}")
            try:
                AgentConfig.load_config(default_config_path)
            except Exception as e:
                self.logger.warning(f"Failed to load config from default path: {e}")
        
        # Ensure configuration is initialized one way or another
        AgentConfig.ensure_initialized()
        
        
        self._dataset_path = AgentConfig.dataset_path()
        self._max_questions = AgentConfig.config().get("max_questions", -1)
        
        
        if not self._dataset_path:
            self.logger.warning("Dataset path is not set in the configuration")
    
    # Define dataset_path as a read-only property
    @property
    def dataset_path(self) -> str:
        """Get the immutable dataset path."""
        return self._dataset_path
    
    def _setup_logger(self):
        """
        Configure the logger with appropriate handlers and formatting.
        """
        # Only configure if the logger doesn't already have handlers
        if not self.logger.handlers:
            # Set an appropriate log level
            self.logger.setLevel(logging.INFO)
            
            # Create console handler that logs to stdout
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Add the handler to the logger
            self.logger.addHandler(console_handler)
            
            # Prevent propagation to root logger to avoid duplicate messages
            self.logger.propagate = False
            
            self.logger.debug("rStarDataLoader logger has been configured")
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load data from the JSON dataset file.
        
        Returns:
            List of problem dictionaries in the format:
            {
                "id": problem_id,
                "question": problem_text,
                "answer": answer_text,
                "solution": "",  # Empty placeholder
                "url": ""        # Empty placeholder
            }
        """
        if not self._dataset_path:
            self.logger.error("Cannot load data: dataset path is not set")
            return []
            
        self.logger.info(f"Loading dataset from: {self._dataset_path}")
        
        # Ensure we have an absolute path
        dataset_path = Path(self._dataset_path)
        
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return []
            
        # Convert to list of dictionaries with the expected format
        problems = []
        for item in data:
            problem_id = str(item.get("index", ""))
            problem = item.get("question", "")
            answer = item.get("answer", "")
            
            # Add to problems list with the expected format
            problems.append({
                "id": problem_id,
                "question": problem,
                "answer": answer,
                "solution": "",  # Empty placeholder
                "url": ""        # Empty placeholder
            })
        
        self.logger.info(f"Loaded {len(problems)} problems from dataset")
        # After loading the problems, filter them if max_questions is set
        if self._max_questions > 0 and len(problems) > self._max_questions:
            self.logger.info(f"Limiting to {self._max_questions} problems from the dataset")
            problems = problems[:self._max_questions]
            self.logger.info(f"Filtered to {len(problems)} problems")

        # Log a few sample problems
        num_samples = min(3, len(problems))
        if num_samples > 0:
            self.logger.info(f"Sample of {num_samples} loaded problems:")
            for i in range(num_samples):
                sample = problems[i]
                # Truncate question if too long for logging
                question_preview = sample["question"][:100] + "..." if len(sample["question"]) > 100 else sample["question"]
                self.logger.info(f"Sample {i+1}: ID={sample['id']}, Question={question_preview}, Answer={sample['answer']}")

        return problems