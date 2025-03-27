import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def create_and_launch_docker_setup():
    """
    Creates the necessary directories, populates them with the correct Dockerfile scripts, 
    creates a docker-compose.yml file, and then launches the services using Docker Compose.
    """

    # Define the root directory for the script
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir / "unravel-ai"  # Assuming "unravel-ai" is the main project directory

    # Define the docker directory
    docker_dir = project_dir / "docker"

    # Create the necessary directories
    docker_app_dir = docker_dir / "app"
    docker_sandbox_dir = docker_dir / "sandbox"
    docker_llm_dir = docker_dir / "llm"
    docker_graph_analyzer_dir = docker_dir / "graph-analyzer"
    docker_frontend_dir = docker_dir / "frontend"

    os.makedirs(docker_app_dir, exist_ok=True)
    os.makedirs(docker_sandbox_dir, exist_ok=True)
    os.makedirs(docker_llm_dir, exist_ok=True)
    os.makedirs(docker_graph_analyzer_dir, exist_ok=True)
    os.makedirs(docker_frontend_dir, exist_ok=True)

    # Populate the Dockerfiles with the content
    def populate_dockerfile(dockerfile_path: Path, dockerfile_content: str):
        try:
            with open(dockerfile_path, "w") as dockerfile:
                dockerfile.write(dockerfile_content)
            logging.info(f"Created and populated: {dockerfile_path}")
        except Exception as e:
            logging.error(f"Error creating or writing to Dockerfile: {e}")
            sys.exit(1)

    # Dockerfile content strings (replace with your actual Dockerfile content)
    dockerfile_app_content = """
# Dockerfile for API Service
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc \\
    curl \\
    git \\
    gnupg \\
    ca-certificates \\
    nodejs \\
    npm \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for sandbox
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Install JS-Beautify
RUN npm install -g js-beautify

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
# Create necessary directories
RUN mkdir -p data/source data/decompiled data/specs data/reconstructed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for web interface
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    dockerfile_sandbox_content = """
# Dockerfile for Sandbox Service
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8050

# Command to run the sandbox manager
CMD ["python", "src/core/sandbox_manager.py"]
"""

    dockerfile_llm_content = """
# Dockerfile for LLM Service
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install huggingface_hub requests numpy psutil setuptools wheel

# Install llama-cpp-python
RUN pip install llama-cpp-python

# Create necessary directories
RUN mkdir -p /app/models

# Copy the LLM interface module
COPY src/core/llm_interface.py /app/llm_interface.py

# Download the model (this might take a while)
# You may need to adjust the MODEL_ID and MODEL_FILE based on your needs
ARG MODEL_ID=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
ARG MODEL_FILE=mistral-7b-instruct-v0.2.Q4_K_M.gguf

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$MODEL_ID', filename='$MODEL_FILE', local_dir='/app/models')"

# Expose port
EXPOSE 8100

# Command to run the LLM service
CMD ["python", "src/services/llm_service.py"]
"""

    dockerfile_graph_analyzer_content = """
# Dockerfile for Code Graph Analyzer
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (if needed)
EXPOSE 80

# Command to run the graph analyzer (adjust as needed)
CMD ["python", "src/core/code_graph_analyzer.py", "-i", "/app/code", "-o", "/app/output"]
"""

    dockerfile_frontend_content = """
# Dockerfile for Web Interface (Frontend)
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .\n
# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx config (if you have one)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \n
    CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

# Start nginx server
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""

    # Populate each Dockerfile
    populate_dockerfile(docker_app_dir / "Dockerfile", dockerfile_app_content)
    populate_dockerfile(docker_sandbox_dir / "Dockerfile", dockerfile_sandbox_content)
    populate_dockerfile(docker_llm_dir / "Dockerfile", dockerfile_llm_content)
    populate_dockerfile(docker_graph_analyzer_dir / "Dockerfile", dockerfile_graph_analyzer_content)
    populate_dockerfile(docker_frontend_dir / "Dockerfile", dockerfile_frontend_content)

    # Change directory to the docker directory
    os.chdir(docker_dir)

    # Create a docker-compose.yml file
    compose_content = """
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: app/Dockerfile
    ports:
      - "8000:8000"
    networks:
      - unravel-network
  sandbox:
    build:
      context: .
      dockerfile: sandbox/Dockerfile
    ports:
      - "8050:8050"
    networks:
      - unravel-network
    depends_on:
      - api
    privileged: true
  llm:
    build:
      context: .
      dockerfile: llm/Dockerfile
    ports:
      - "8100:8100"
    networks:
      - unravel-network
  graph_analyzer:
    build:
      context: .
      dockerfile: graph-analyzer/Dockerfile
    ports:
      - "8200:8200"
    networks:
      - unravel-network
    depends_on:
      - api
      - llm
  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "80:80"
    networks:
      - unravel-network

networks:
  unravel-network:
    name: unravel-network
"""
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)

    # Launch the services using Docker Compose
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        logging.info("Successfully launched services using Docker Compose.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error launching services: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_and_launch_docker_setup()
