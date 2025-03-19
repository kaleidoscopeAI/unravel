import os
import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import PostgresDsn, field_validator, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    PROJECT_NAME: str = "Unravel AI"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    ASYNC_DATABASE_URL: Optional[str] = None
    
    # 1. By the time validate_settings is called, DATABASE_URL is already set
    # 2. We'll use DATABASE_URL to construct ASYNC_DATABASE_URL
    # 3. If DATABASE_URL is None, Settings will attempt to construct it from these parts
    POSTGRES_SERVER: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    
    # Model settings
    MODEL_PATH: str = "/app/models/mistral7b_onnx"
    TOKENIZER_PATH: str = "/app/models/mistral7b_tokenizer"
    
    # File storage settings
    UPLOAD_DIR: str = "/app/uploads"
    
    # Sandbox settings
    SANDBOX_ENABLED: bool = True
    SANDBOX_TIMEOUT: int = 3600  # 1 hour
    
    # LLM API settings
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Authentication settings
    USERS_OPEN_REGISTRATION: bool = False
    
    @field_validator("ASYNC_DATABASE_URL")
    def construct_async_db_url(cls, v: Optional[str], info: Dict[str, Any]) -> str:
        """Construct an async database URL from a sync one."""
        if v is not None:
            return v
        
        db_url = info.data.get("DATABASE_URL")
        if db_url:
            # Replace 'postgresql://' with 'postgresql+asyncpg://'
            if db_url.startswith("postgresql://"):
                return db_url.replace("postgresql://", "postgresql+asyncpg://")
            elif db_url.startswith("postgres://"):
                return db_url.replace("postgres://", "postgresql+asyncpg://")
        
        # If DATABASE_URL is None, construct it from components
        postgres_params = {
            "user": info.data.get("POSTGRES_USER"),
            "password": info.data.get("POSTGRES_PASSWORD"),
            "host": info.data.get("POSTGRES_SERVER"),
            "db": info.data.get("POSTGRES_DB"),
        }
        
        if all(postgres_params.values()):
            dsn = PostgresDsn.build(
                scheme="postgresql+asyncpg",
                user=postgres_params["user"],
                password=postgres_params["password"],
                host=postgres_params["host"],
                path=f"/{postgres_params['db']}"
            )
            return str(dsn)
        
        return None
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a global settings instance
settings = Settings()
