"""
Logging utilities for the AI Research Agent.

This module provides functions for setting up and configuring logging.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from core.utils.path_utils import get_active_run_timestamp


def setup_logging(run_dir=None, level=logging.INFO, log_name=None, capture_docker=True):
    """
    Set up logging with console and file handlers.
    
    Args:
        run_dir (str, optional): Directory where logs should be stored. 
                                 If None, logs will only be output to console.
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_name (str, optional): Name for the log file. 
                                 If None, a timestamped name will be used.
        capture_docker (bool): Whether to capture Docker logs. Defaults to True.
                                 
    Returns:
        logging.Logger: Configured logger object.
    """
    # Format for log messages
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create root logger
    root_logger = logging.getLogger('ai_research_agent')
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplication
    while root_logger.handlers:
        root_logger.handlers.pop()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if run_dir is provided
    if run_dir:
        # Create logs directory if it doesn't exist
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename
        if log_name is None:
            try:
                timestamp = get_active_run_timestamp()
            except Exception:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_name = f"experiment_{timestamp}.log"
        
        # Create file handler
        log_file_path = log_dir / log_name
        file_handler = logging.FileHandler(str(log_file_path))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Log file created at: {log_file_path}")
        
        # Capture Docker container logs if requested
        # Instead of trying to use Docker CLI, just capture stdout/stderr
        import sys
        
        class StreamToLogger:
            def __init__(self, logger, log_level=logging.INFO):
                self.logger = logger
                self.log_level = log_level
                self.linebuf = ''

            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.log_level, line.rstrip())
                
            def flush(self):
                pass
        
        # Capture stdout and stderr
        sys.stdout = StreamToLogger(root_logger, logging.INFO)
        sys.stderr = StreamToLogger(root_logger, logging.ERROR)
    
    # Make sure handlers propagate to child loggers
    root_logger.propagate = False
    
    # Log initialization message
    root_logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    
    return root_logger

def get_logger(name=None):
    """
    Get a logger with the given name.
    
    Args:
        name (str, optional): Name for the logger. If None, returns the root ai_research_agent logger.
        
    Returns:
        logging.Logger: Logger object.
    """
    if not name:
        return logging.getLogger('ai_research_agent')
    
    # For child loggers, use the full namespace
    return logging.getLogger(f'ai_research_agent.{name}')