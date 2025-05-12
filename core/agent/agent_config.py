"""
Centralized configuration manager for AI Research Agent components.

Provides a class-based singleton configuration registry to share configuration values 
across components without relying on environment variables, ensuring immutability once initialized.
"""

import yaml
import json
import logging
import os
from typing import Dict, Any, Optional, ClassVar
from pathlib import Path

logger = logging.getLogger("ai_research_agent.config")

class AgentConfig:
    """
    Class-based singleton configuration registry that enforces immutability after initialization.
    
    This implementation stores configuration values as class attributes (static variables),
    making them accessible across all instances and modules that import this class.
    """
    # Class variables for configuration storage
    _config_file: ClassVar[Optional[Path]] = None
    _dataset_path: ClassVar[Optional[str]] = None
    _reasoner_model: ClassVar[Optional[str]] = None
    _batch_size: ClassVar[int] = 4  # Default batch size
    _max_num_predict: ClassVar[int] = 2048  # Default max tokens to generate
    _full_config: ClassVar[Dict[str, Any]] = {}
    _initialized: ClassVar[bool] = False
    _locked: ClassVar[bool] = False
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and update class state.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the loaded configuration
        """
        if cls._locked:
            raise RuntimeError("Configuration is locked and cannot be modified")
        
        config_path = Path(config_path)
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Store the config file path and full config dictionary
            cls._config_file = config_path
            cls._full_config = config.copy()
            
            # Extract specific values of interest
            cls._dataset_path = config.get("dataset_path")
            cls._reasoner_model = config.get("reasoner_model")
            cls._batch_size = config.get("batch_size", 4)  # Default to 4 if not specified
            cls._max_num_predict = config.get("max_num_predict", 2048)  # Default to 2048 if not specified
            
            # Write config to a file that can be accessed by subprocesses
            cls._write_config_to_file()
            
            cls._initialized = True
            logger.info(f"Loaded configuration with dataset_path={cls._dataset_path}, "
                        f"reasoner_model={cls._reasoner_model}, "
                        f"batch_size={cls._batch_size}, "
                        f"max_num_predict={cls._max_num_predict}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    @classmethod
    def _write_config_to_file(cls) -> None:
        """Write the configuration to a JSON file for subprocess access."""
        try:
            config_dir = Path("/experiment")
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "agent_config.json"
            
            config_data = {
                "dataset_path": cls._dataset_path,
                "reasoner_model": cls._reasoner_model,
                "batch_size": cls._batch_size,
                "max_num_predict": cls._max_num_predict,
                "full_config": cls._full_config
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Configuration written to {config_path}")
        except Exception as e:
            logger.error(f"Error writing configuration file: {e}")
    
    @classmethod
    def _load_from_file(cls) -> bool:
        """
        Load configuration from JSON file if it exists.
        Returns True if successful, False otherwise.
        """
        try:
            config_path = Path("/experiment/agent_config.json")
            if not config_path.exists():
                logger.warning("Configuration file not found")
                return False
                
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            cls._dataset_path = config_data.get("dataset_path")
            cls._reasoner_model = config_data.get("reasoner_model")
            cls._batch_size = config_data.get("batch_size", 4)
            cls._max_num_predict = config_data.get("max_num_predict", 2048)
            cls._full_config = config_data.get("full_config", {})
            cls._initialized = True
            
            logger.info(f"Loaded configuration from file with dataset_path={cls._dataset_path}, "
                        f"reasoner_model={cls._reasoner_model}, "
                        f"batch_size={cls._batch_size}, "
                        f"max_num_predict={cls._max_num_predict}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
            return False
    
    @classmethod
    def ensure_initialized(cls) -> None:
        """
        Ensure the configuration is initialized by checking various sources.
        This should be called at the start of any component that uses the configuration.
        """
        if not cls._initialized:
            # Try to load from file first
            if cls._load_from_file():
                return
                
            # Look for environment variables as fallback
            env_model = os.environ.get("REASONER_MODEL")
            env_dataset = os.environ.get("DATASET_PATH")
            env_batch_size = os.environ.get("BATCH_SIZE")
            env_max_num_predict = os.environ.get("MAX_NUM_PREDICT")
            
            if any([env_model, env_dataset, env_batch_size, env_max_num_predict]):
                cls._reasoner_model = env_model
                cls._dataset_path = env_dataset
                
                if env_batch_size:
                    try:
                        cls._batch_size = int(env_batch_size)
                    except ValueError:
                        logger.warning(f"Invalid BATCH_SIZE environment variable: {env_batch_size}, using default: 4")
                
                if env_max_num_predict:
                    try:
                        cls._max_num_predict = int(env_max_num_predict)
                    except ValueError:
                        logger.warning(f"Invalid MAX_NUM_PREDICT environment variable: {env_max_num_predict}, using default: 2048")
                
                cls._initialized = True
                logger.info(f"Loaded configuration from environment variables: "
                            f"dataset_path={cls._dataset_path}, "
                            f"reasoner_model={cls._reasoner_model}, "
                            f"batch_size={cls._batch_size}, "
                            f"max_num_predict={cls._max_num_predict}")
                return
                
            logger.warning("Could not initialize configuration from any source")
    
    @classmethod
    def dataset_path(cls) -> Optional[str]:
        """Get the absolute dataset path from configuration."""
        cls.ensure_initialized()
        
        if cls._dataset_path is None:
            logger.error("dataset_path is not set in configuration")
            return None
            
        # Return absolute path
        if not Path(cls._dataset_path).is_absolute():
            # Remove 'data/' prefix if it exists
            path_str = cls._dataset_path
            if path_str.startswith("data/"):
                path_str = path_str[5:]  # Remove the 'data/' prefix
            
            # Add /data prefix
            return str(Path("/data") / path_str)
        
        return cls._dataset_path
    
    @classmethod
    def reasoner_model(cls) -> Optional[str]:
        """Get the reasoner model from configuration."""
        cls.ensure_initialized()
        return cls._reasoner_model
    
    @classmethod
    def batch_size(cls) -> int:
        """Get the batch size for parallel processing."""
        cls.ensure_initialized()
        return cls._batch_size
    
    @classmethod
    def max_num_predict(cls) -> int:
        """Get the maximum number of tokens to generate per request."""
        cls.ensure_initialized()
        return cls._max_num_predict
    
    @classmethod
    def config_file(cls) -> Optional[Path]:
        """Get the path to the config file."""
        cls.ensure_initialized()
        return cls._config_file
    
    @classmethod
    def config(cls) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        cls.ensure_initialized()
        return cls._full_config.copy()  # Return a copy to prevent modification
    
    @classmethod
    def lock(cls) -> None:
        """Lock the configuration to prevent further modifications."""
        cls._locked = True
        logger.info("Configuration is now locked")
    
    @classmethod
    def unlock(cls) -> None:
        """Unlock the configuration to allow modifications (use with caution)."""
        cls._locked = False
        logger.info("Configuration is now unlocked")