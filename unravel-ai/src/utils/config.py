"""
Unravel AI - Configuration Module
Manages configuration settings from environment variables and config files
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for Unravel AI"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_data = {}
        
        # Load environment variables
        self._load_env_vars()
        
        # Load configuration file if provided
        if config_path:
            self._load_config_file(config_path)
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        # Database settings
        self.config_data["DATABASE_URL"] = os.environ.get("DATABASE_URL", "")
        
        # API settings
        self.config_data["API_HOST"] = os.environ.get("API_HOST", "0.0.0.0")
        self.config_data["API_PORT"] = int(os.environ.get("API_PORT", "8000"))
        self.config_data["API_DEBUG"] = os.environ.get("API_DEBUG", "False").lower() == "true"
        
        # Security settings
        self.config_data["SECRET_KEY"] = os.environ.get("SECRET_KEY", "")
        self.config_data["JWT_ALGORITHM"] = os.environ.get("JWT_ALGORITHM", "HS256")
        self.config_data["ACCESS_TOKEN_EXPIRE_MINUTES"] = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # File storage settings
        self.config_data["STORAGE_TYPE"] = os.environ.get("STORAGE_TYPE", "local")
        self.config_data["STORAGE_PATH"] = os.environ.get("STORAGE_PATH", "./data")
        self.config_data["AWS_S3_BUCKET"] = os.environ.get("AWS_S3_BUCKET", "")
        
        # Sandbox settings
        self.config_data["SANDBOX_ENABLED"] = os.environ.get("SANDBOX_ENABLED", "True").lower() == "true"
        self.config_data["SANDBOX_TIMEOUT"] = int(os.environ.get("SANDBOX_TIMEOUT", "3600"))
        
        # LLM settings
        self.config_data["LLM_PROVIDER"] = os.environ.get("LLM_PROVIDER", "openai")
        self.config_data["LLM_API_KEY"] = os.environ.get("LLM_API_KEY", "")
        self.config_data["LLM_MODEL"] = os.environ.get("LLM_MODEL", "gpt-4")
    
    def _load_config_file(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Update config with file values
                self.config_data.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self.config_data.copy()

# Global configuration instance
config = Config()

def initialize_config(config_path: Optional[str] = None):
    """
    Initialize global configuration
    
    Args:
        config_path: Path to configuration file
    """
    global config
    config = Config(config_path)
    
    # Validate required settings
    if not config.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set")
    
    if not config.get("SECRET_KEY"):
        logger.warning("SECRET_KEY not set, using a random key")
        import secrets
        config.set("SECRET_KEY", secrets.token_hex(32))
    
    return config
