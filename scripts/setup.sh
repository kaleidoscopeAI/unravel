#!/bin/bash
# Unravel AI Setup Script
# This script sets up the core infrastructure for Unravel AI on Render

set -e

echo "==============================================="
echo "Unravel AI - Core Infrastructure Setup"
echo "==============================================="

# Configuration
POSTGRES_URL="postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com/unravel_ai"
REPO_NAME="unravel-ai"
WORK_DIR="$(pwd)/$REPO_NAME"

# Create project directory structure
echo "Creating project directory structure..."
mkdir -p $WORK_DIR/{src,config,scripts,docker,tests}
mkdir -p $WORK_DIR/src/{core,api,utils,data,sandbox,licensing}
mkdir -p $WORK_DIR/config/{dev,prod}
mkdir -p $WORK_DIR/docker/{app,sandbox}

# Initialize Git repository
echo "Initializing Git repository..."
cd $WORK_DIR
git init

# Create database initialization script
echo "Creating database initialization script..."
cat > $WORK_DIR/scripts/db_init.py << 'EOL'
#!/usr/bin/env python3
"""
Unravel AI - Database Initialization Script
Initialize the PostgreSQL database schema for Unravel AI
"""

import os
import sys
import psycopg2
import argparse
import logging
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# SQL Scripts for database initialization
SQL_CREATE_TABLES = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    company VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    account_type VARCHAR(20) NOT NULL DEFAULT 'free',
    api_key VARCHAR(64) UNIQUE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Software table to store ingested software
CREATE TABLE IF NOT EXISTS software (
    software_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_type VARCHAR(50),
    original_file_path TEXT,
    work_file_path TEXT,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Decompiled files table
CREATE TABLE IF NOT EXISTS decompiled_files (
    file_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Specification files table
CREATE TABLE IF NOT EXISTS spec_files (
    spec_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reconstructed software table
CREATE TABLE IF NOT EXISTS reconstructed_software (
    reconstructed_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    directory_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Mimicked software table
CREATE TABLE IF NOT EXISTS mimicked_software (
    mimicked_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    target_language VARCHAR(50) NOT NULL,
    directory_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Licenses table
CREATE TABLE IF NOT EXISTS licenses (
    license_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    software_id VARCHAR(36) REFERENCES software(software_id),
    license_type VARCHAR(50) NOT NULL,
    license_key TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expiration_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Sandbox executions table
CREATE TABLE IF NOT EXISTS sandbox_executions (
    execution_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id),
    user_id VARCHAR(36) REFERENCES users(user_id),
    container_id VARCHAR(100),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL,
    log_path TEXT
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    usage_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    endpoint VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    status_code INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    ip_address VARCHAR(45)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_software_user_id ON software(user_id);
CREATE INDEX IF NOT EXISTS idx_decompiled_files_software_id ON decompiled_files(software_id);
CREATE INDEX IF NOT EXISTS idx_spec_files_software_id ON spec_files(software_id);
CREATE INDEX IF NOT EXISTS idx_reconstructed_software_id ON reconstructed_software(software_id);
CREATE INDEX IF NOT EXISTS idx_mimicked_software_id ON mimicked_software(software_id);
CREATE INDEX IF NOT EXISTS idx_licenses_user_id ON licenses(user_id);
CREATE INDEX IF NOT EXISTS idx_licenses_software_id ON licenses(software_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_executions_user_id ON sandbox_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
"""

def init_database(db_url):
    """Initialize the database schema"""
    logger.info("Connecting to database...")
    
    try:
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        logger.info("Creating tables...")
        cursor.execute(SQL_CREATE_TABLES)
        
        # Create admin user if it doesn't exist
        cursor.execute("""
        INSERT INTO users (user_id, email, password_hash, account_type)
        VALUES ('admin', 'admin@unravelai.com', '$2b$12$IWGGSEr9r5PXqV7vn9bOXe6tgkr55BPh8sgXaqWJ5lMpnJZ0yc5b6', 'admin')
        ON CONFLICT (email) DO NOTHING;
        """)
        
        logger.info("Database initialization completed successfully.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Unravel AI database")
    parser.add_argument("--db-url", help="PostgreSQL connection URL", 
                        default=os.environ.get("DATABASE_URL"))
    
    args = parser.parse_args()
    
    if not args.db_url:
        logger.error("Database URL not provided. Use --db-url or set DATABASE_URL environment variable.")
        sys.exit(1)
    
    init_database(args.db_url)
EOL

# Create Docker configuration
echo "Creating Docker configuration..."
cat > $WORK_DIR/docker/app/Dockerfile << 'EOL'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (needed for sandbox management)
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/source /app/data/decompiled /app/data/specs /app/data/reconstructed

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Create requirements.txt
echo "Creating requirements.txt..."
cat > $WORK_DIR/requirements.txt << 'EOL'
# Core dependencies
fastapi>=0.95.0
uvicorn>=0.21.1
pydantic>=2.0.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
psycopg2-binary>=2.9.6
asyncpg>=0.27.0
sqlalchemy>=2.0.0
alembic>=1.11.0

# Authentication and security
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.0.1
cryptography>=40.0.0

# AWS integration (for file storage)
boto3>=1.26.0

# Docker integration (for sandbox)
docker>=6.1.0
dockerpty>=0.4.1

# Utilities
tiktoken>=0.3.0
tenacity>=8.2.0
backoff>=2.2.1
networkx>=3.1
numpy>=1.24.0
aiohttp>=3.8.4

# Async utilities
asyncio>=3.4.3
aiofiles>=23.1.0

# Testing
pytest>=7.3.1
pytest-asyncio>=0.21.0
httpx>=0.24.0
EOL

# Create core configuration module
echo "Creating core configuration module..."
cat > $WORK_DIR/src/utils/config.py << 'EOL'
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
EOL

# Create main API module
echo "Creating main API module..."
cat > $WORK_DIR/src/api/main.py << 'EOL'
"""
Unravel AI - Main API Module
FastAPI application for Unravel AI
"""

import os
import sys
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import internal modules
from src.utils.config import initialize_config, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize configuration
initialize_config()

# Create FastAPI app
app = FastAPI(
    title="Unravel AI API",
    description="API for Unravel AI software ingestion and mimicry system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Unravel AI API", "status": "running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# API version endpoint
@app.get("/version")
async def version():
    """Get API version"""
    return {"version": "1.0.0"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

# Include API routers
# These will be implemented in other files

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.get("API_HOST", "0.0.0.0"),
        port=config.get("API_PORT", 8000),
        reload=config.get("API_DEBUG", False)
    )
EOL

# Create deployment script
echo "Creating deployment script for Render..."
cat > $WORK_DIR/scripts/deploy_render.py << 'EOL'
#!/usr/bin/env python3
"""
Unravel AI - Render Deployment Script
Automates the deployment of Unravel AI to Render.com
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import requests
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class RenderDeployer:
    """Handles deployment to Render.com"""
    
    def __init__(self, api_key: str):
        """
        Initialize the deployer
        
        Args:
            api_key: Render API key
        """
        self.api_key = api_key
        self.api_base_url = "https://api.render.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_web_service(self, name: str, repo_url: str, branch: str = "main",
                          env_vars: Dict[str, str] = None, plan: str = "starter") -> Dict[str, Any]:
        """
        Create a new web service on Render
        
        Args:
            name: Service name
            repo_url: GitHub repository URL
            branch: Repository branch
            env_vars: Environment variables
            plan: Render plan (starter, standard, etc.)
            
        Returns:
            Response from Render API
        """
        endpoint = f"{self.api_base_url}/services"
        
        payload = {
            "type": "web_service",
            "name": name,
            "env": "python",
            "plan": plan,
            "region": "ohio",
            "branch": branch,
            "repo": repo_url,
            "autoDeploy": "yes",
            "envVars": [{"key": k, "value": v} for k, v in (env_vars or {}).items()]
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code != 201:
            logger.error(f"Failed to create web service: {response.text}")
            return None
        
        return response.json()
    
    def create_database(self, name: str, plan: str = "starter") -> Dict[str, Any]:
        """
        Create a new PostgreSQL database on Render
        
        Args:
            name: Database name
            plan: Render plan
            
        Returns:
            Response from Render API
        """
        endpoint = f"{self.api_base_url}/databases"
        
        payload = {
            "name": name,
            "engine": "postgres",
            "version": "16",
            "region": "ohio",
            "plan": plan,
            "ipAllowList": [
                {
                    "source": "0.0.0.0/0",
                    "description": "everywhere"
                }
            ]
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code != 201:
            logger.error(f"Failed to create database: {response.text}")
            return None
        
        return response.json()
    
    def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """
        Get service deployment status
        
        Args:
            service_id: Render service ID
            
        Returns:
            Service status
        """
        endpoint = f"{self.api_base_url}/services/{service_id}"
        
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to get service status: {response.text}")
            return None
        
        return response.json()
    
    def trigger_deploy(self, service_id: str) -> bool:
        """
        Trigger a manual deployment
        
        Args:
            service_id: Render service ID
            
        Returns:
            Success status
        """
        endpoint = f"{self.api_base_url}/services/{service_id}/deploys"
        
        response = requests.post(endpoint, headers=self.headers)
        
        if response.status_code != 201:
            logger.error(f"Failed to trigger deployment: {response.text}")
            return False
        
        logger.info(f"Deployment triggered for service {service_id}")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Unravel AI to Render.com")
    parser.add_argument("--api-key", required=True, help="Render API key")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--branch", default="main", help="Repository branch")
    parser.add_argument("--env-file", help="Path to environment variables file (.env)")
    parser.add_argument("--service-name", default="unravel-ai", help="Service name")
    parser.add_argument("--db-name", default="unravel-ai-db", help="Database name")
    parser.add_argument("--plan", default="starter", help="Render plan")
    
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = {}
    if args.env_file:
        try:
            with open(args.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        except Exception as e:
            logger.error(f"Failed to load environment variables: {str(e)}")
            sys.exit(1)
    
    # Initialize deployer
    deployer = RenderDeployer(args.api_key)
    
    # Create database
    logger.info(f"Creating database {args.db_name}...")
    db_result = deployer.create_database(args.db_name, args.plan)
    
    if not db_result:
        logger.error("Failed to create database")
        sys.exit(1)
    
    db_id = db_result["id"]
    logger.info(f"Database created with ID: {db_id}")
    
    # Wait for database to be created (in a real script, add actual waiting logic)
    logger.info("Waiting for database to be provisioned... (this may take a few minutes)")
    
    # Add database URL to environment variables
    # Note: In a real implementation, you would get this from the Render API
    # For now, we'll use a placeholder
    env_vars["DATABASE_URL"] = f"${{RENDER_DATABASE_URL}}"
    
    # Create web service
    logger.info(f"Creating web service {args.service_name}...")
    service_result = deployer.create_web_service(
        args.service_name,
        args.repo_url,
        args.branch,
        env_vars,
        args.plan
    )
    
    if not service_result:
        logger.error("Failed to create web service")
        sys.exit(1)
    
    service_id = service_result["id"]
    logger.info(f"Web service created with ID: {service_id}")
    
    logger.info("Deployment initiated. Monitor the status on your Render dashboard.")

if __name__ == "__main__":
    main()
EOL

# Create environment file
echo "Creating environment file..."
cat > $WORK_DIR/.env.example << 'EOL'
# Unravel AI Environment Configuration

# Database settings
DATABASE_URL=postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com/unravel_ai

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Security settings
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File storage settings
STORAGE_TYPE=local
STORAGE_PATH=./data
# If using S3 storage:
# STORAGE_TYPE=s3
# AWS_S3_BUCKET=your-bucket-name
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key

# Sandbox settings
SANDBOX_ENABLED=True
SANDBOX_TIMEOUT=3600  # 1 hour in seconds

# LLM settings
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4
EOL

# Create Render YAML configuration
echo "Creating Render YAML configuration..."
cat > $WORK_DIR/render.yaml << 'EOL'
# Unravel AI Render Configuration
services:
  # Web API Service
  - type: web
    name: unravel-ai
    env: python
    region: ohio
    plan: starter
    branch: main
    healthCheckPath: /health
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: DATABASE_URL
        fromDatabase:
          name: unravel-ai-db
          property: connectionString
      - key: SECRET_KEY
        generateValue: true
      - key: STORAGE_TYPE
        value: local
      - key: STORAGE_PATH
        value: /data
      - key: SANDBOX_ENABLED
        value: "true"
      - key: LLM_PROVIDER
        value: openai
      - key: LLM_MODEL
        value: gpt-4

# Database
databases:
  - name: unravel-ai-db
    databaseName: unravel_ai
    user: unravel_admin
    plan: starter
    region: ohio
    ipAllowList:
      - source: 0.0.0.0/0
        description: everywhere
EOL

# Create main readme
echo "Creating README.md..."
cat > $WORK_DIR/README.md << 'EOL'
# Unravel AI

An intelligent software ingestion and mimicry system capable of analyzing, decompiling, and reconstructing software with enhanced capabilities.

## Features

- **Software Ingestion**: Decompile and analyze binaries and obfuscated code
- **Specification Generation**: Create detailed specifications from analyzed code
- **Software Reconstruction**: Generate enhanced versions of ingested software
- **Language Mimicry**: Create new software in different programming languages
- **Secure Sandbox**: Test generated applications in a secure environment
- **Licensing System**: Manage licenses and monetization

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL database
- API keys for LLM services (OpenAI, Anthropic, etc.)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/unravel-ai.git
   cd unravel-ai
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create configuration:
   ```
   cp .env.example .env
   # Edit .env with your settings
   ```

5. Initialize the database:
   ```
   python scripts/db_init.py --db-url <your-database-url>
   ```

6. Start the API server:
   ```
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Deployment

### Deploying to Render

1. Fork this repository to your GitHub account

2. Create a new Web Service on Render, linking to your forked repo

3. Set the required environment variables in the Render dashboard

4. Alternatively, use our deployment script:
   ```
   python scripts/deploy_render.py --api-key YOUR_RENDER_API_KEY --repo-url YOUR_GITHUB_REPO
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOL

# Make scripts executable
chmod +x $WORK_DIR/scripts/db_init.py
chmod +x $WORK_DIR/scripts/deploy_render.py

# Create setup script for the project
echo "Creating project setup script..."
cat > $WORK_DIR/setup.sh << 'EOL'
#!/bin/bash
# Unravel AI Setup Script

set -e

# Check for required dependencies
echo "Checking dependencies..."
DEPS=("python3" "pip" "virtualenv" "git" "docker")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
  if ! command -v $dep &> /dev/null; then
    MISSING_DEPS+=($dep)
  fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
  echo "Error: Missing dependencies: ${MISSING_DEPS[@]}"
  echo "Please install these dependencies and try again."
  exit 1
fi

# Set up virtual environment
echo "Setting up virtual environment..."
python3 -m virtualenv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set up environment
if [ ! -f .env ]; then
  echo "Creating .env file..."
  cp .env.example .env
  echo "Please edit .env file with your settings."
fi

# Initialize the database if DATABASE_URL is set
if grep -q "DATABASE_URL" .env; then
  DB_URL=$(grep "DATABASE_URL" .env | cut -d '=' -f2-)
  if [ ! -z "$DB_URL" ]; then
    echo "Initializing database..."
    python scripts/db_init.py --db-url "$DB_URL"
  fi
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/{source,decompiled,specs,reconstructed}

echo "Setup complete!"
echo "To start the API server, run: uvicorn src.api.main:app --reload"
EOL

chmod +x $WORK_DIR/setup.sh

# Create a core module for file type detection
echo "Creating core detection module..."
cat > $WORK_DIR/src/core/detection.py << 'EOL'
"""
Unravel AI - File Type Detection Module
Detects file types and programming languages
"""

import os
import re
import magic
import logging
import subprocess
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class FileType(Enum):
    """File types supported by Unravel AI"""
    BINARY = "binary"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    PHP = "php"
    RUBY = "ruby"
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

class FileDetector:
    """Detects file types and programming languages"""
    
    def __init__(self):
        """Initialize the file detector"""
        self.mime = magic.Magic(mime=True)
        
        # File extension mappings
        self.ext_map = {
            # Binaries
            ".exe": FileType.BINARY,
            ".dll": FileType.BINARY,
            ".so": FileType.BINARY,
            ".dylib": FileType.BINARY,
            ".bin": FileType.BINARY,
            ".o": FileType.BINARY,
            ".obj": FileType.BINARY,
            
            # Scripts and source files
            ".js": FileType.JAVASCRIPT,
            ".mjs": FileType.JAVASCRIPT,
            ".ts": FileType.TYPESCRIPT,
            ".tsx": FileType.TYPESCRIPT,
            ".jsx": FileType.JAVASCRIPT,
            ".py": FileType.PYTHON,
            ".cpp": FileType.CPP,
            ".cc": FileType.CPP,
            ".cxx": FileType.CPP,
            ".c": FileType.C,
            ".h": FileType.C,
            ".hpp": FileType.CPP,
            ".cs": FileType.CSHARP,
            ".java": FileType.JAVA,
            ".go": FileType.GO,
            ".rs": FileType.RUST,
            ".swift": FileType.SWIFT,
            ".kt": FileType.KOTLIN,
            ".php": FileType.PHP,
            ".rb": FileType.RUBY,
            ".asm": FileType.ASSEMBLY,
            ".s": FileType.ASSEMBLY
        }
        
        # MIME type mappings
        self.mime_map = {
            "application/x-executable": FileType.BINARY,
            "application/x-sharedlib": FileType.BINARY,
            "application/x-mach-binary": FileType.BINARY,
            "application/x-dosexec": FileType.BINARY,
            "application/x-object": FileType.BINARY,
            
            "text/javascript": FileType.JAVASCRIPT,
            "application/javascript": FileType.JAVASCRIPT,
            "text/x-python": FileType.PYTHON,
            "text/x-c": FileType.C,
            "text/x-c++": FileType.CPP,
            "text/x-java": FileType.JAVA,
            "text/x-csharp": FileType.CSHARP,
            "text/x-go": FileType.GO,
            "text/x-rust": FileType.RUST,
            "text/x-swift": FileType.SWIFT,
            "text/x-kotlin": FileType.KOTLIN,
            "text/x-php": FileType.PHP,
            "text/x-ruby": FileType.RUBY,
            "text/x-asm": FileType.ASSEMBLY
        }
    
    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect the type of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        # Check file extension first
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.ext_map:
            return self.ext_map[ext]
        
        # Try to detect by MIME type
        try:
            mime_type = self.mime.from_file(file_path)
            
            # Look for known MIME types
            for mime_pattern, file_type in self.mime_map.items():
                if mime_pattern in mime_type:
                    return file_type
            
            # Check if binary or text
            if "text/" in mime_type:
                # Try to infer from content
                return self._infer_from_content(file_path)
            elif "application/" in mime_type or "binary" in mime_type:
                return FileType.BINARY
        except Exception as e:
            logger.error(f"Error detecting MIME type: {str(e)}")
        
        # Try using the 'file' command
        try:
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            # Check output for clues
            lower_output = output.lower()
            
            if any(x in lower_output for x in ["elf", "executable", "binary", "mach-o", "pe32"]):
                return FileType.BINARY
            elif "javascript" in lower_output:
                return FileType.JAVASCRIPT
            elif "python" in lower_output:
                return FileType.PYTHON
            elif "c++ source" in lower_output:
                return FileType.CPP
            elif "c source" in lower_output:
                return FileType.C
            elif "java source" in lower_output:
                return FileType.JAVA
            elif "c#" in lower_output:
                return FileType.CSHARP
            elif "go source" in lower_output:
                return FileType.GO
            elif "rust" in lower_output:
                return FileType.RUST
            elif "swift source" in lower_output:
                return FileType.SWIFT
            elif "php script" in lower_output:
                return FileType.PHP
            elif "ruby script" in lower_output:
                return FileType.RUBY
            elif "assembler source" in lower_output:
                return FileType.ASSEMBLY
        except Exception as e:
            logger.error(f"Error running 'file' command: {str(e)}")
        
        # If all else fails, try to infer from content
        return self._infer_from_content(file_path)
    
    def _infer_from_content(self, file_path: str) -> FileType:
        """
        Infer file type from content
        
        Args:
            file_path: Path to the file
            
        Returns:
            Inferred file type
        """
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(4096)  # Read first 4K of the file
                
                # Look for language indicators
                if re.search(r'import\s+{.*?}\s+from|require\(|export\s+default|=>|function\s+\w+\s*\(', content):
                    return FileType.JAVASCRIPT
                elif re.search(r'interface\s+\w+|class\s+\w+\s+implements|<\w+>|:.*?;', content):
                    return FileType.TYPESCRIPT
                elif re.search(r'import\s+\w+|def\s+\w+\s*\(.*\):|class\s+\w+\s*:', content):
                    return FileType.PYTHON
                elif re.search(r'#include\s+<\w+\.h>|template\s+<typename|std::', content):
                    return FileType.CPP
                elif re.search(r'#include\s+<\w+\.h>|void\s+\w+\s*\(|int\s+main\s*\(', content):
                    return FileType.C
                elif re.search(r'public\s+class|public\s+static\s+void\s+main|@Override', content):
                    return FileType.JAVA
                elif re.search(r'namespace\s+\w+|public\s+class\s+\w+\s*:|using\s+System;', content):
                    return FileType.CSHARP
                elif re.search(r'package\s+main|func\s+\w+\s*\(|import\s+\(', content):
                    return FileType.GO
                elif re.search(r'fn\s+\w+\s*\(|impl\s+\w+|use\s+std::', content):
                    return FileType.RUST
                elif re.search(r'import\s+Foundation|@objc|class\s+\w+\s*:\s*\w+', content):
                    return FileType.SWIFT
                elif re.search(r'<?php|namespace\s+\w+;|\$\w+\s*=', content):
                    return FileType.PHP
                elif re.search(r'require\s+[\'"]\w+[\'"]|def\s+\w+\s*(\(|$)|class\s+\w+\s*<', content):
                    return FileType.RUBY
                elif re.search(r'\.globl|\.section|\.text', content):
                    return FileType.ASSEMBLY
        except Exception as e:
            logger.error(f"Error inferring file type from content: {str(e)}")
        
        # Default to unknown
        return FileType.UNKNOWN
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a file and return detailed information
        
        Args:
            file_path: Path to the file
            
        Returns:
            File analysis information
        """
        file_type = self.detect_file_type(file_path)
        
        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": file_type.value,
            "file_size": os.path.getsize(file_path),
            "mime_type": self.mime.from_file(file_path),
            "last_modified": os.path.getmtime(file_path)
        }
        
        # Add file-type specific analysis
        if file_type == FileType.BINARY:
            result.update(self._analyze_binary(file_path))
        elif file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT]:
            result.update(self._analyze_javascript(file_path))
        elif file_type == FileType.PYTHON:
            result.update(self._analyze_python(file_path))
        elif file_type in [FileType.C, FileType.CPP]:
            result.update(self._analyze_c_cpp(file_path))
        
        return result
    
    def _analyze_binary(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze binary file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary analysis info
        """
        result = {
            "is_executable": os.access(file_path, os.X_OK),
            "architecture": "unknown",
            "format": "unknown"
        }
        
        try:
            # Try to get more info using 'file' command
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            # Parse architecture and format
            if "ELF" in output:
                result["format"] = "ELF"
                if "x86-64" in output:
                    result["architecture"] = "x86_64"
                elif "80386" in output:
                    result["architecture"] = "x86"
                elif "ARM" in output:
                    result["architecture"] = "ARM"
            elif "PE32" in output:
                result["format"] = "PE"
                if "x86-64" in output:
                    result["architecture"] = "x86_64"
                else:
                    result["architecture"] = "x86"
            elif "Mach-O" in output:
                result["format"] = "Mach-O"
                if "x86_64" in output:
                    result["architecture"] = "x86_64"
                elif "arm64" in output:
                    result["architecture"] = "ARM64"
        except Exception as e:
            logger.error(f"Error analyzing binary: {str(e)}")
        
        return result
    
    def _analyze_javascript(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript file
        
        Args:
            file_path: Path to the file
            
        Returns:
            JavaScript analysis info
        """
        result = {
            "imports": [],
            "exports": [],
            "functions": [],
            "classes": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract imports
                import_patterns = [
                    r'import\s+{(.*?)}\s+from\s+[\'"](.+?)[\'"]',
                    r'import\s+(\w+)\s+from\s+[\'"](.+?)[\'"]',
                    r'require\s*\(\s*[\'"](.+?)[\'"]\s*\)'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        if len(match.groups()) == 2:
                            result["imports"].append({
                                "imported": match.group(1),
                                "source": match.group(2)
                            })
                        elif len(match.groups()) == 1:
                            result["imports"].append({
                                "source": match.group(1)
                            })
                
                # Extract functions
                for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:function|\(.*?\)\s*=>)|\()', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
                # Extract exports
                for match in re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', content):
                    result["exports"].append(match.group(1))
                
        except Exception as e:
            logger.error(f"Error analyzing JavaScript: {str(e)}")
        
        return result
    
    def _analyze_python(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze Python file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Python analysis info
        """
        result = {
            "imports": [],
            "functions": [],
            "classes": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract imports
                import_patterns = [
                    r'import\s+(\w+)',
                    r'from\s+(\w+(?:\.\w+)*)\s+import\s+(.*)'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        if len(match.groups()) == 1:
                            result["imports"].append({
                                "module": match.group(1)
                            })
                        elif len(match.groups()) == 2:
                            result["imports"].append({
                                "module": match.group(1),
                                "imports": [x.strip() for x in match.group(2).split(',')]
                            })
                
                # Extract functions
                for match in re.finditer(r'def\s+(\w+)\s*\(', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
        except Exception as e:
            logger.error(f"Error analyzing Python: {str(e)}")
        
        return result
    
    def _analyze_c_cpp(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze C/C++ file
        
        Args:
            file_path: Path to the file
            
        Returns:
            C/C++ analysis info
        """
        result = {
            "includes": [],
            "functions": [],
            "classes": [],
            "preprocessor": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract includes
                for match in re.finditer(r'#include\s+[<"](.+?)[>"]', content):
                    result["includes"].append(match.group(1))
                
                # Extract functions
                for match in re.finditer(r'(?:int|void|bool|char|double|float|auto|std::string|string)\s+(\w+)\s*\(', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
                # Extract preprocessor directives
                for match in re.finditer(r'#define\s+(\w+)', content):
                    result["preprocessor"].append({
                        "type": "define",
                        "name": match.group(1)
                    })
                
        except Exception as e:
            logger.error(f"Error analyzing C/C++: {str(e)}")
        
        return result
EOL

# Create sandbox module
echo "Creating sandbox module..."
cat > $WORK_DIR/src/sandbox/manager.py << 'EOL'
"""
Unravel AI - Sandbox Execution Manager
Manages Docker-based sandboxes for secure application execution
"""

import os
import time
import json
import uuid
import logging
import docker
import tarfile
import io
import subprocess
import socket
import threading
import webbrowser
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    app_dir: str
    app_name: str
    app_type: str
    language: str
    framework: str
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    network_enabled: bool = True
    expose_ports: List[int] = field(default_factory=list)
    mount_volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 7200  # 2 hours default timeout
    auto_open_browser: bool = True
    container_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with defaults based on app type"""
        import re
        import secrets
        
        if not self.container_name:
            # Generate a unique container name
            safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '', self.app_name.lower())
            self.container_name = f"unravel-{safe_name}-{secrets.token_hex(4)}"
        
        # Set default ports based on app type and framework
        if not self.expose_ports:
            if self.app_type == "web":
                if self.framework == "flask":
                    self.expose_ports = [5000]
                elif self.framework == "django":
                    self.expose_ports = [8000]
                elif self.framework == "express":
                    self.expose_ports = [3000]
                elif self.framework == "react":
                    self.expose_ports = [3000]
                else:
                    self.expose_ports = [8080]
            elif self.app_type == "api":
                self.expose_ports = [8000]
            elif self.app_type == "database":
                if "postgres" in self.framework.lower():
                    self.expose_ports = [5432]
                elif "mysql" in self.framework.lower() or "mariadb" in self.framework.lower():
                    self.expose_ports = [3306]
                elif "mongo" in self.framework.lower():
                    self.expose_ports = [27017]
                else:
                    self.expose_ports = [8080]

class DockerSandbox:
    """Executes applications in a Docker sandbox"""
    
    def __init__(self):
        """Initialize the Docker sandbox"""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise RuntimeError(f"Docker initialization failed: {str(e)}")
        
        self.running_containers = {}
        self.ports_mapping = {}
    
    def _build_image(self, config: SandboxConfig) -> str:
        """
        Build a Docker image for the app
        
        Args:
            config: Sandbox configuration
            
        Returns:
            Image ID
        """
        logger.info(f"Building Docker image for {config.app_name}")
        
        # Create a Dockerfile based on app type
        dockerfile_content = self._generate_dockerfile(config)
        
        # Create a Docker context tarball
        context_tar = io.BytesIO()
        with tarfile.open(fileobj=context_tar, mode='w') as tar:
            # Add Dockerfile
            dockerfile_info = tarfile.TarInfo("Dockerfile")
            dockerfile_bytes = dockerfile_content.encode('utf-8')
            dockerfile_info.size = len(dockerfile_bytes)
            tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))
            
            # Add application files
            self._add_dir_to_tar(tar, config.app_dir, arcname=".")
        
        # Reset tarball file pointer
        context_tar.seek(0)
        
        # Build the image
        image, build_logs = self.client.images.build(
            fileobj=context_tar,
            custom_context=True,
            tag=f"{config.container_name}:latest",
            pull=True,
            rm=True
        )
        
        # Log build process
        for log in build_logs:
            if 'stream' in log:
                log_line = log['stream'].strip()
                if log_line:
                    logger.debug(f"Build: {log_line}")
        
        logger.info(f"Built Docker image: {image.id}")
        return image.id
    
    def _add_dir_to_tar(self, tar: tarfile.TarFile, source_dir: str, arcname: str) -> None:
        """
        Add a directory to a tarball
        
        Args:
            tar: Tarfile object
            source_dir: Source directory path
            arcname: Archive name (path in tarball)
        """
        for root, dirs, files in os.walk(source_dir):
            # Calculate path in archive
            archive_root = os.path.join(arcname, os.path.relpath(root, source_dir))
            
            for file in files:
                # Skip any compiled Python files, virtual environments, or hidden files
                if file.endswith('.pyc') or '__pycache__' in root or '.git' in root:
                    continue
                if file.startswith('.') and file != '.env':
                    continue
                    
                file_path = os.path.join(root, file)
                archive_path = os.path.join(archive_root, file)
                
                # Add file to tarball
                tar.add(file_path, arcname=archive_path, recursive=False)
    
    def _generate_dockerfile(self, config: SandboxConfig) -> str:
        """
        Generate a Dockerfile for the app
        
        Args:
            config: Sandbox configuration
            
        Returns:
            Dockerfile content
        """
        language = config.language.lower()
        framework = config.framework.lower()
        
        dockerfile = "# Generated Dockerfile\n"
        
        # Base image selection
        if language == "python":
            dockerfile += "FROM python:3.9-slim\n\n"
            
            # Install system dependencies
            dockerfile += "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            dockerfile += "    build-essential gcc \\\n"
            dockerfile += "    && rm -rf /var/lib/apt/lists/*\n\n"
            
            # Set working directory
            dockerfile += "WORKDIR /app\n\n"
            
            # Copy requirements first for better caching
            dockerfile += "COPY requirements.txt /app/requirements.txt\n"
            dockerfile += "RUN pip install --no-cache-dir -r requirements.txt\n\n"
            
            # Copy app code
            dockerfile += "COPY . /app/\n\n"
            
            # Default command based on framework
            if framework == "flask":
                dockerfile += "ENV FLASK_APP=app.py\n"
                dockerfile += "ENV FLASK_ENV=development\n"
                dockerfile += "EXPOSE 5000\n"
                dockerfile += 'CMD ["flask", "run", "--host=0.0.0.0"]\n'
            elif framework == "django":
                dockerfile += "EXPOSE 8000\n"
                # Run migrations and start server
                dockerfile += 'CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]\n'
            else:
                # Generic Python command
                dockerfile += "EXPOSE 8000\n"
                dockerfile += 'CMD ["python", "app.py"]\n'
                
        elif language == "javascript" or language == "typescript":
            if framework == "react":
                dockerfile += "FROM node:16-alpine\n\n"
                dockerfile += "WORKDIR /app\n\n"
                dockerfile += "COPY package*.json ./\n"
                dockerfile += "RUN npm install\n\n"
                dockerfile += "COPY . .\n\n"
                dockerfile += "EXPOSE 3000\n"
                dockerfile += 'CMD ["npm", "start"]\n'
            else:  # Express or other Node.js frameworks
                dockerfile += "FROM node:16-alpine\n\n"
                dockerfile += "WORKDIR /app\n\n"
                dockerfile += "COPY package*.json ./\n"
                dockerfile += "RUN npm install\n\n"
                dockerfile += "COPY . .\n\n"
                dockerfile += "EXPOSE 3000\n"
                dockerfile += 'CMD ["node", "index.js"]\n'
                
        elif language == "java":
            dockerfile += "FROM openjdk:11-jdk-slim\n\n"
            dockerfile += "WORKDIR /app\n\n"
            
            # If it's a Maven project
            if os.path.exists(os.path.join(config.app_dir, "pom.xml")):
                dockerfile += "COPY pom.xml .\n"
                dockerfile += "RUN apt-get update && apt-get install -y maven\n"
                dockerfile += "RUN mvn dependency:go-offline\n\n"
                dockerfile += "COPY . .\n"
                dockerfile += "RUN mvn package\n\n"
                dockerfile += "EXPOSE 8080\n"
                dockerfile += 'CMD ["java", "-jar", "target/*.jar"]\n'
            # If it's a Gradle project
            elif os.path.exists(os.path.join(config.app_dir, "build.gradle")):
                dockerfile += "COPY build.gradle .\n"
                dockerfile += "RUN apt-get update && apt-get install -y gradle\n"
                dockerfile += "RUN gradle dependencies\n\n"
                dockerfile += "COPY . .\n"
                dockerfile += "RUN gradle build\n\n"
                dockerfile += "EXPOSE 8080\n"
                dockerfile += 'CMD ["java", "-jar", "build/libs/*.jar"]\n'
            else:
                dockerfile += "COPY . .\n"
                dockerfile += "RUN javac *.java\n\n"
                dockerfile += 'CMD ["java", "Main"]\n'
                
        elif language == "go":
            dockerfile += "FROM golang:1.17-alpine\n\n"
            dockerfile += "WORKDIR /app\n\n"
            dockerfile += "COPY go.* ./\n"
            dockerfile += "RUN go mod download\n\n"
            dockerfile += "COPY . .\n"
            dockerfile += "RUN go build -o main .\n\n"
            dockerfile += "EXPOSE 8080\n"
            dockerfile += 'CMD ["./main"]\n'
            
        else:
            # Generic fallback
            dockerfile += "FROM ubuntu:20.04\n\n"
            dockerfile += "WORKDIR /app\n\n"
            dockerfile += "COPY . .\n\n"
            dockerfile += 'CMD ["bash", "start.sh"]\n'
        
        return dockerfile
    
    def run_sandbox(self, config: SandboxConfig) -> Dict[str, Any]:
        """
        Run the app in a sandbox
        
        Args:
            config: Sandbox configuration
            
        Returns:
            Sandbox information including URLs
        """
        try:
            # Build Docker image
            image_id = self._build_image(config)
            
            # Prepare port bindings
            port_bindings = {}
            for port in config.expose_ports:
                # Find a free port on host
                free_port = self._find_free_port()
                port_bindings[port] = free_port
                self.ports_mapping[config.container_name] = port_bindings
            
            # Prepare volume mounts
            volumes = {}
            for host_path, container_path in config.mount_volumes.items():
                volumes[os.path.abspath(host_path)] = {'bind': container_path, 'mode': 'rw'}
            
            # Prepare resource constraints
            resource_constraints = {
                'mem_limit': config.memory_limit,
                'cpu_quota': int(100000 * config.cpu_limit),  # Docker uses CPU quota in microseconds
                'cpu_period': 100000
            }
            
            # Start the container
            logger.info(f"Starting container: {config.container_name}")
            container = self.client.containers.run(
                image=image_id,
                name=config.container_name,
                detach=True,
                ports=port_bindings,
                volumes=volumes,
                environment=config.environment,
                network_mode='bridge' if config.network_enabled else 'none',
                **resource_constraints
            )
            
            # Store container reference
            self.running_containers[config.container_name] = container
            
            # Wait for container to be ready
            ready = self._wait_for_container_ready(container, port_bindings, timeout=30)
            
            # Generate access URLs
            urls = {}
            for container_port, host_port in port_bindings.items():
                protocol = "http"
                urls[container_port] = f"{protocol}://localhost:{host_port}"
            
            # Open browser if requested
            if config.auto_open_browser and urls and ready:
                primary_url = list(urls.values())[0]
                logger.info(f"Opening browser: {primary_url}")
                webbrowser.open(primary_url)
            
            # Set up container monitoring
            self._start_container_monitoring(config.container_name, config.timeout_seconds)
            
            # Return sandbox info
            return {
                "container_id": container.id,
                "container_name": config.container_name,
                "status": "running" if ready else "starting",
                "ports": port_bindings,
                "urls": urls,
                "ready": ready
            }
            
        except Exception as e:
            logger.error(f"Failed to run sandbox: {str(e)}")
            # Clean up if there was an error
            self.cleanup_sandbox(config.container_name)
            raise RuntimeError(f"Sandbox execution failed: {str(e)}")
    
    def _find_free_port(self) -> int:
        """
        Find a free port on the host
        
        Returns:
            Available port number
        """
        # Create a temporary socket to find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    def _wait_for_container_ready(
        self, 
        container: docker.models.containers.Container, 
        port_bindings: Dict[int, int],
        timeout: int = 30
    ) -> bool:
        """
        Wait for container to be ready by checking port availability
        
        Args:
            container: Docker container
            port_bindings: Port mappings
            timeout: Timeout in seconds
            
        Returns:
            Whether container is ready
        """
        logger.info(f"Waiting for container to be ready (timeout: {timeout}s)")
        
        # Simple readiness check - just wait a moment
        time.sleep(2)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if container is still running
            container.reload()
            if container.status != "running":
                logger.error(f"Container failed to start: {container.status}")
                return False
            
            # Try to connect to the first mapped port
            if port_bindings:
                container_port = list(port_bindings.keys())[0]
                host_port = port_bindings[container_port]
                
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        result = s.connect_ex(('localhost', host_port))
                        if result == 0:
                            logger.info(f"Container is ready (port {host_port} is open)")
                            return True
                except Exception:
                    pass
            
            # Wait before next check
            time.sleep(1)
        
        logger.warning(f"Container readiness check timed out after {timeout}s")
        return False
    
    def _start_container_monitoring(self, container_name: str, timeout_seconds: int) -> None:
        """
        Start monitoring thread for the container
        
        Args:
            container_name: Container name
            timeout_seconds: Timeout in seconds
        """
        def monitor_container():
            logger.info(f"Starting container monitoring thread for {container_name}")
            
            # Record start time
            start_time = time.time()
            
            while True:
                try:
                    # Sleep to avoid tight loop
                    time.sleep(5)
                    
                    # Check if container still exists
                    if container_name not in self.running_containers:
                        logger.info(f"Container {container_name} no longer exists, stopping monitoring")
                        break
                    
                    # Get container
                    container = self.running_containers[container_name]
                    
                    # Reload container info
                    container.reload()
                    
                    # Check container status
                    if container.status != "running":
                        logger.info(f"Container {container_name} is no longer running (status: {container.status})")
                        self.cleanup_sandbox(container_name)
                        break
                    
                    # Check timeout
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    if elapsed_time > timeout_seconds:
                        logger.info(f"Container {container_name} reached timeout ({timeout_seconds}s), stopping")
                        self.cleanup_sandbox(container_name)
                        break
                    
                except Exception as e:
                    logger.error(f"Error in container monitoring: {str(e)}")
                    # Continue monitoring despite errors
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=monitor_container)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    
    def get_container_logs(self, container_name: str) -> str:
        """
        Get container logs
        
        Args:
            container_name: Container name
            
        Returns:
            Container logs
        """
        try:
            if container_name in self.running_containers:
                container = self.running_containers[container_name]
                logs = container.logs().decode('utf-8')
                return logs
            else:
                return "Container not found"
        except Exception as e:
            logger.error(f"Failed to get container logs: {str(e)}")
            return f"Error retrieving logs: {str(e)}"
    
    def cleanup_sandbox(self, container_name: str) -> bool:
        """
        Clean up sandbox resources
        
        Args:
            container_name: Container name
            
        Returns:
            Success status
        """
        logger.info(f"Cleaning up sandbox: {container_name}")
        
        try:
            # Check if container exists
            if container_name in self.running_containers:
                container = self.running_containers[container_name]
                
                # Stop container
                try:
                    container.stop(timeout=10)
                    logger.info(f"Stopped container: {container_name}")
                except Exception as e:
                    logger.error(f"Error stopping container: {str(e)}")
                
                # Remove container
                try:
                    container.remove(force=True)
                    logger.info(f"Removed container: {container_name}")
                except Exception as e:
                    logger.error(f"Error removing container: {str(e)}")
                
                # Remove from running containers
                del self.running_containers[container_name]
            
            # Remove ports mapping
            if container_name in self.ports_mapping:
                del self.ports_mapping[container_name]
            
            # Try to remove the image
            try:
                self.client.images.remove(f"{container_name}:latest")
                logger.info(f"Removed image: {container_name}:latest")
            except Exception as e:
                logger.error(f"Error removing image: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {str(e)}")
            return False
    
    def cleanup_all(self) -> None:
        """Clean up all sandbox resources"""
        logger.info("Cleaning up all sandboxes")
        
        # Make a copy of container names to avoid dictionary size change during iteration
        container_names = list(self.running_containers.keys())
        
        for container_name in container_names:
            self.cleanup_sandbox(container_name)

class SandboxManager:
    """Manages sandbox environments"""
    
    def __init__(self):
        """Initialize the sandbox manager"""
        self.docker_sandbox = DockerSandbox()
        self.active_sandboxes = {}
    
    def create_sandbox(self, app_dir: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a sandbox environment
        
        Args:
            app_dir: Application directory
            app_config: Application configuration
            
        Returns:
            Sandbox information
        """
        logger.info(f"Creating sandbox for app: {app_config['name']}")
        
        # Create sandbox configuration
        config = SandboxConfig(
            app_dir=app_dir,
            app_name=app_config['name'],
            app_type=app_config['type'],
            language=app_config['language'],
            framework=app_config['framework'],
            memory_limit=app_config.get('memory_limit', '2g'),
            cpu_limit=app_config.get('cpu_limit', 1.0),
            network_enabled=app_config.get('network_enabled', True),
            expose_ports=app_config.get('expose_ports', []),
            environment=app_config.get('environment', {}),
            timeout_seconds=app_config.get('timeout_seconds', 7200),
            auto_open_browser=app_config.get('auto_open_browser', True)
        )
        
        # Run the sandbox
        result = self.docker_sandbox.run_sandbox(config)
        
        # Store active sandbox info
        self.active_sandboxes[config.container_name] = {
            "config": config,
            "info": result,
            "start_time": time.time()
        }
        
        return result
    
    def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Get sandbox status
        
        Args:
            sandbox_id: Sandbox ID (container name)
            
        Returns:
            Sandbox status
        """
        if sandbox_id not in self.active_sandboxes:
            return {"status": "not_found"}
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        # Refresh container info
        try:
            container = self.docker_sandbox.running_containers.get(sandbox_id)
            if container:
                container.reload()
                status = container.status
            else:
                status = "removed"
        except Exception:
            status = "error"
        
        # Get ports and URLs
        ports = self.docker_sandbox.ports_mapping.get(sandbox_id, {})
        urls = {}
        for container_port, host_port in ports.items():
            urls[container_port] = f"http://localhost:{host_port}"
        
        return {
            "id": sandbox_id,
            "status": status,
            "start_time": sandbox["start_time"],
            "elapsed_time": time.time() - sandbox["start_time"],
            "ports": ports,
            "urls": urls
        }
    
    def get_sandbox_logs(self, sandbox_id: str) -> str:
        """
        Get sandbox logs
        
        Args:
            sandbox_id: Sandbox ID (container name)
            
        Returns:
            Sandbox logs
        """
        if sandbox_id not in self.active_sandboxes:
            return "Sandbox not found"
        
        return self.docker_sandbox.get_container_logs(sandbox_id)
    
    def stop_sandbox(self, sandbox_id: str) -> bool:
        """
        Stop a sandbox
        
        Args:
            sandbox_id: Sandbox ID (container name)
            
        Returns:
            Success status
        """
        if sandbox_id not in self.active_sandboxes:
            return False
        
        success = self.docker_sandbox.cleanup_sandbox(sandbox_id)
        
        # Remove from active sandboxes if cleanup was successful
        if success:
            del self.active_sandboxes[sandbox_id]
        
        return success
    
    def stop_all_sandboxes(self) -> None:
        """Stop all active sandboxes"""
        logger.info(f"Stopping all sandboxes ({len(self.active_sandboxes)} active)")
        
        # Make a copy of sandbox IDs to avoid dictionary size change during iteration
        sandbox_ids = list(self.active_sandboxes.keys())
        
        for sandbox_id in sandbox_ids:
            self.stop_sandbox(sandbox_id)
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        self.stop_all_sandboxes()
        self.docker_sandbox.cleanup_all()
EOL

# Create complete project launch script
echo "Creating launch script..."
cat > $WORK_DIR/launch.sh << 'EOL'
#!/bin/bash
# Unravel AI - Launch Script

set -e

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Check if virtualenv is activated
if [ -z "$VIRTUAL_ENV" ]; then
  # Check if venv directory exists
  if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
  else
    echo "Warning: Virtual environment not found. Create one with './setup.sh'"
  fi
fi

# Function to check if required environment variables are set
check_env_vars() {
  local missing=false
  
  if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL is not set"
    missing=true
  fi
  
  if [ -z "$SECRET_KEY" ]; then
    echo "Warning: SECRET_KEY is not set. Generating a random one..."
    export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
  fi
  
  if [ "$missing" = true ]; then
    echo "Please set the required environment variables or create a .env file"
    exit 1
  fi
}

# Function to initialize database
init_database() {
  echo "Initializing database..."
  python scripts/db_init.py --db-url "$DATABASE_URL"
}

# Function to start the API server
start_api() {
  local port=${API_PORT:-8000}
  local host=${API_HOST:-0.0.0.0}
  local reload=${API_DEBUG:-False}
  
  echo "Starting API server on $host:$port..."
  
  if [ "$reload" = "True" ] || [ "$reload" = "true" ]; then
    echo "Running in debug mode with auto-reload..."
    uvicorn src.api.main:app --host $host --port $port --reload
  else
    uvicorn src.api.main:app --host $host --port $port
  fi
}

# Main entry point
case "$1" in
  setup)
    ./setup.sh
    ;;
  init-db)
    check_env_vars
    init_database
    ;;
  start)
    check_env_vars
    start_api
    ;;
  deploy)
    if [ -z "$2" ]; then
      echo "Usage: $0 deploy <api_key>"
      exit 1
    fi
    
    echo "Deploying to Render..."
    python scripts/deploy_render.py --api-key "$2" --repo-url "$(git remote get-url origin)" --env-file .env
    ;;
  *)
    echo "Unravel AI - Launch Script"
    echo "Usage: $0 [setup|init-db|start|deploy]"
    echo ""
    echo "Commands:"
    echo "  setup    - Set up the project (create virtual environment, install dependencies)"
    echo "  init-db  - Initialize the database"
    echo "  start    - Start the API server"
    echo "  deploy   - Deploy to Render (requires Render API key)"
    exit 1
    ;;
esac

exit 0
EOL

chmod +x $WORK_DIR/launch.sh

# Create migration script for session state
echo "Creating migration script..."
cat > $WORK_DIR/scripts/migrate_session.py << 'EOL'
#!/usr/bin/env python3
"""
Unravel AI - Session Migration Tool
Exports and imports session state between conversations for continuous development
"""

import os
import sys
import json
import argparse
import logging
import base64
import zlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session state for Unravel AI development"""
    
    def __init__(self, session_dir: str = ".sessions"):
        """
        Initialize the session manager
        
        Args:
            session_dir: Directory to store session data
        """
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
    
    def export_session(self, name: str, data: Dict[str, Any]) -> str:
        """
        Export session data
        
        Args:
            name: Session name
            data: Session data
            
        Returns:
            Session file path
        """
        session_path = os.path.join(self.session_dir, f"{name}.json")
        
        # Save the session data
        with open(session_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Create compressed version for easier sharing
        compressed_data = zlib.compress(json.dumps(data).encode('utf-8'))
        compressed_b64 = base64.b64encode(compressed_data).decode('utf-8')
        
        compressed_path = os.path.join(self.session_dir, f"{name}.b64")
        with open(compressed_path, 'w') as f:
            f.write(compressed_b64)
        
        logger.info(f"Exported session to {session_path}")
        logger.info(f"Compressed session to {compressed_path}")
        
        return session_path
    
    def import_session(self, path_or_data: str) -> Dict[str, Any]:
        """
        Import session data
        
        Args:
            path_or_data: Path to session file or compressed data
            
        Returns:
            Session data
        """
        # Check if input is a file path
        if os.path.exists(path_or_data):
            logger.info(f"Importing session from file: {path_or_data}")
            
            # Determine file type
            if path_or_data.endswith('.json'):
                with open(path_or_data, 'r') as f:
                    data = json.load(f)
            elif path_or_data.endswith('.b64'):
                with open(path_or_data, 'r') as f:
                    compressed_b64 = f.read()
                compressed_data = base64.b64decode(compressed_b64)
                data_str = zlib.decompress(compressed_data).decode('utf-8')
                data = json.loads(data_str)
            else:
                logger.error("Unsupported file format")
                raise ValueError("Unsupported file format")
        else:
            # Try to interpret as compressed data
            try:
                compressed_data = base64.b64decode(path_or_data)
                data_str = zlib.decompress(compressed_data).decode('utf-8')
                data = json.loads(data_str)
                logger.info("Imported session from compressed data")
            except Exception as e:
                logger.error(f"Failed to import session: {str(e)}")
                raise ValueError("Invalid session data")
        
        return data
    
    def list_sessions(self) -> List[str]:
        """
        List available sessions
        
        Returns:
            List of session names
        """
        sessions = []
        
        for file in os.listdir(self.session_dir):
            if file.endswith('.json'):
                sessions.append(file[:-5])  # Remove .json extension
        
        return sessions

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unravel AI Session Migration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export session")
    export_parser.add_argument("name", help="Session name")
    export_parser.add_argument("--file", help="JSON file to export (default: stdin)")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import session")
    import_parser.add_argument("path", help="Path to session file or compressed data")
    
    # List command
    subparsers.add_parser("list", help="List available sessions")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.command == "export":
        # Read data from file or stdin
        if args.file:
            with open(args.file, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(sys.stdin.read())
        
        # Export session
        session_path = manager.export_session(args.name, data)
        print(f"Session exported to {session_path}")
        
    elif args.command == "import":
        # Import session
        data = manager.import_session(args.path)
        
        # Output data to stdout
        print(json.dumps(data, indent=2))
        
    elif args.command == "list":
        # List sessions
        sessions = manager.list_sessions()
        
        if sessions:
            print("Available sessions:")
            for session in sessions:
                print(f"  {session}")
        else:
            print("No sessions available")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
EOL

chmod +x $WORK_DIR/scripts/migrate_session.py

# Generate phase transition token
TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)

# Create transition readme for next phase
cat > $WORK_DIR/NEXT_PHASE.md << EOL
# Unravel AI - Phase Transition

Use the following token to continue work in the next conversation:

\`\`\`
UNRAVEL_PHASE_TOKEN=${TOKEN}
\`\`\`

## Phase 1 Completion

Phase 1 has established the core infrastructure:

1. Project structure for a modular Python-based system
2. Database schema and initialization script
3. Configuration management system
4. Docker-based sandbox environment
5. FastAPI-based API foundation
6. Deployment configuration for Render

## Phase 2 Tasks

1. Core engine implementation migrated from Kaleidoscope
2. Implement the software ingestion pipeline
3. Create LLM integration for analysis
4. Build database interaction layer

## How to Continue

1. Run ./setup.sh to initialize the environment
2. Use the phase transition token in the next conversation
3. Begin implementation of Phase 2 components
EOL

# Create .gitignore
cat > $WORK_DIR/.gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE and editor files
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Logs
*.log
logs/

# Data files
data/*
!data/.gitkeep

# Environment variables
.env

# Session data
.sessions/

# Docker
.docker/

# Sandbox files
sandbox/
/db_init.py
chmod +x $WORK_DIR/scripts