import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Config:
    # Base configuration
    BASE_DIR = Path(__file__).resolve().parent
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a/unravel_ai")
    
    # File storage
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB
    
    # JWT Authentication
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-for-unravel-ai-change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
    
    # Docker Sandbox
    DOCKER_SANDBOX_MEMORY_LIMIT = "2g"
    DOCKER_SANDBOX_CPU_LIMIT = 1.0
    DOCKER_SANDBOX_TIMEOUT = 3600  # 1 hour
    
    # LLM Service
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    # Paths for decompilers
    RADARE2_PATH = os.getenv("RADARE2_PATH", "r2")
    GHIDRA_PATH = os.getenv("GHIDRA_PATH", "/opt/ghidra")
    RETDEC_PATH = os.getenv("RETDEC_PATH", "retdec-decompiler")
    
    # Work directories
    WORK_DIR = os.path.join(BASE_DIR, "workdir")
    DECOMPILED_DIR = os.path.join(WORK_DIR, "decompiled")
    SPECS_DIR = os.path.join(WORK_DIR, "specs")
    RECONSTRUCTED_DIR = os.path.join(WORK_DIR, "reconstructed")
    MIMICRY_DIR = os.path.join(WORK_DIR, "mimicry")
    
    # Make sure directories exist
    for dir_path in [UPLOAD_FOLDER, WORK_DIR, DECOMPILED_DIR, SPECS_DIR, RECONSTRUCTED_DIR, MIMICRY_DIR]:
        os.makedirs(dir_path, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    # In production, we'd set different security settings
    # Use environment variables for sensitive information

def get_config():
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()

config = get_config()
