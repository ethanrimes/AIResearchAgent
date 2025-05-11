import json
import logging
from typing import Dict, List, Any
import os
import sys

class rStarDataLoader:
    """
    Simple data loader for math problem datasets in JSON format.
    """
    def __init__(self):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the JSON dataset file
            logger: Optional logger instance (will create one if not provided)
        """
        self.dataset_path = os.environ.get("DATASET_PATH")
        
        # Set up module-specific logger
        self.logger = logging.getLogger("ai_research_agent.rstar_data_loader")
        self._setup_logger()
    
    def _setup_logger(self):
        """
        Configure the logger with appropriate handlers and formatting.
        """
        # Only configure if the logger doesn't already have handlers
        if not self.logger.handlers:
            # Set an appropriate log level
            self.logger.setLevel(logging.INFO)
            
            # Create console handler that logs to stderr
            console_handler = logging.StreamHandler(sys.stderr)
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
        self.dataset_path = os.environ.get("DATASET_PATH")
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        
        if not os.path.isabs(self.dataset_path):
            self.dataset_path = os.path.join("/app", self.dataset_path)
        
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
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