"""
Configuration management for the application.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """Application configuration."""
    
    # Default paths
    DEFAULT_MODEL_DIR = "models"
    DEFAULT_DATA_DIR = "data"
    
    # Model configuration
    DEFAULT_MODEL_NAME = "mlp_model"
    DEFAULT_HIDDEN_LAYERS = [64, 32]
    DEFAULT_ACTIVATION = "relu"
    DEFAULT_OPTIMIZER = "adam"
    DEFAULT_LOSS = "mse"
    
    # Training configuration
    DEFAULT_EPOCHS = 500
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_VALIDATION_SPLIT = 0.0
    
    # Optimization configuration
    DEFAULT_Vc_BOUNDS = (80, 170)
    DEFAULT_fn_BOUNDS = (0.08, 0.17)
    DEFAULT_OPTIMIZATION_METHOD = "COBYLA"
    DEFAULT_MAX_ITERATIONS = 5000
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON config file (optional)
        """
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.config = {
            'model_dir': self.DEFAULT_MODEL_DIR,
            'model_name': self.DEFAULT_MODEL_NAME,
            'hidden_layers': self.DEFAULT_HIDDEN_LAYERS,
            'activation': self.DEFAULT_ACTIVATION,
            'optimizer': self.DEFAULT_OPTIMIZER,
            'loss': self.DEFAULT_LOSS,
            'epochs': self.DEFAULT_EPOCHS,
            'batch_size': self.DEFAULT_BATCH_SIZE,
            'validation_split': self.DEFAULT_VALIDATION_SPLIT,
            'Vc_bounds': self.DEFAULT_Vc_BOUNDS,
            'fn_bounds': self.DEFAULT_fn_BOUNDS,
            'optimization_method': self.DEFAULT_OPTIMIZATION_METHOD,
            'max_iterations': self.DEFAULT_MAX_ITERATIONS
        }
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        self._set_defaults()
        self.config.update(file_config)
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file."""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like assignment."""
        self.config[key] = value


def get_config(config_file: str = None) -> Config:
    """
    Get configuration instance.
    
    Args:
        config_file: Path to config file (optional)
        
    Returns:
        Config instance
    """
    return Config(config_file)

