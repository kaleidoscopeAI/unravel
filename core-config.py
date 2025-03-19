"""
Configuration utilities for Kaleidoscope AI
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default configuration
default_config = {
    "STORAGE_PATH": os.getenv("STORAGE_PATH", "./data"),
    "MAX_UPLOAD_SIZE": 100 * 1024 * 1024,  # 100 MB
    "ALLOWED_EXTENSIONS": [
        "py", "js", "c", "cpp", "cs", "java", "asm", "s", 
        "exe", "dll", "so", "dylib"
    ],
    "LLM_ENDPOINT": os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1"),
    "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4"),
    "SANDBOX_TIMEOUT": int(os.getenv("SANDBOX_TIMEOUT", "7200")),  # 2 hours
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "4")),
    "DEBUG": os.getenv("DEBUG", "false").lower() == "true"
}

# Try to load config from file
config_path = os.getenv("CONFIG_PATH", "config.json")
config = default_config.copy()

try:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            # Update config with loaded values
            config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_path}")
except Exception as e:
    logger.warning(f"Error loading configuration from {config_path}: {str(e)}")
    logger.info("Using default configuration")

# Create storage directories
storage_path = Path(config["STORAGE_PATH"])
for subdir in ["source", "decompiled", "specs", "reconstructed"]:
    (storage_path / subdir).mkdir(parents=True, exist_ok=True)

def get(key: str, default: Any = None) -> Any:
    """
    Get configuration value
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return config.get(key, default)

def update(key: str, value: Any) -> None:
    """
    Update configuration value
    
    Args:
        key: Configuration key
        value: New value
    """
    config[key] = value

def save() -> None:
    """Save configuration to file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
