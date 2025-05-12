"""
Centralized configuration manager for AI Research Agent components.

Provides a singleton configuration registry to share configuration values across components
without relying on environment variables, ensuring immutability once initialized.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("ai_research_agent.config")

class AgentConfig:
    """
    Singleton configuration registry that enforces immutability after initialization.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Initialize with None values
            self._dataset_path = None
            self._reasoner_model = None
            self._config_file = None
            self._full_config = {}
            self._locked = False
            self._initialized = True
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and update internal state.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the loaded configuration
        """
        if self._locked:
            raise RuntimeError("Configuration is locked and cannot be modified")
        
        config_path = Path(config_path)
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Store the config file path and full config dictionary
            self._config_file = config_path
            self._full_config = config.copy()
            
            # Extract specific values of interest
            self._dataset_path = config.get("dataset_path")
            self._reasoner_model = config.get("reasoner_model")
            
            logger.info(f"Loaded configuration with dataset_path={self._dataset_path}, reasoner_model={self._reasoner_model}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    @property
    def dataset_path(self) -> Optional[str]:
        """Get the absolute dataset path from configuration."""
        if self._dataset_path is None:
            logger.warning("dataset_path is not set in configuration")
            return None
            
        # Return absolute path
        if not Path(self._dataset_path).is_absolute():
            return str(Path("/data") / self._dataset_path)
        return self._dataset_path
    
    @property
    def reasoner_model(self) -> Optional[str]:
        """Get the reasoner model from configuration."""
        return self._reasoner_model
    
    @property
    def config_file(self) -> Optional[Path]:
        """Get the path to the config file."""
        return self._config_file
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._full_config.copy()  # Return a copy to prevent modification
    
    def lock(self) -> None:
        """Lock the configuration to prevent further modifications."""
        self._locked = True
        logger.info("Configuration is now locked")
    
    def unlock(self) -> None:
        """Unlock the configuration to allow modifications (use with caution)."""
        self._locked = False
        logger.info("Configuration is now unlocked")