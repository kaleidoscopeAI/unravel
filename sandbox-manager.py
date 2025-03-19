#!/usr/bin/env python3
"""
Kaleidoscope AI - Sandbox Manager Service
=========================================
A secure, isolated environment for testing generated applications with
self-healing capabilities, resource monitoring, and network isolation.
"""

import os
import sys
import time
import json
import logging
import asyncio
import tempfile
import uuid
import shutil
import tarfile
import io
import re
import platform
import signal
import secrets
import threading
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

import asyncpg
import docker
import requests
import psutil
import aiohttp
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Configuration from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
SANDBOX_MAX_CONTAINERS = int(os.environ.get("SANDBOX_MAX_CONTAINERS", "10"))
SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "3600"))
DEFAULT_MEMORY_LIMIT = os.environ.get("DEFAULT_MEMORY_LIMIT", "2g")
DEFAULT_CPU_LIMIT = float(os.environ.get("DEFAULT_CPU_LIMIT", "1.0"))

class SandboxConfig(BaseModel):
    """Sandbox configuration"""
    sandbox_id: str
    app_dir: str
    app_name: str
    app_type: str
    language: str
    framework: str
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: float = DEFAULT_CPU_LIMIT
    network_enabled: bool = True
    expose_ports: List[int] = Field(default_factory=list)
    mount_volumes: Dict[str, str] = Field(default_factory=dict)
    environment: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS
    auto_open_browser: bool = False
    container_name: Optional[str] = None

class DockerSandbox:
    """Executes applications in a Docker sandbox"""
    
    def __init__(self):
        """Initialize the Docker sandbox"""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized")
            
            # Create isolated network for sandboxes
            self._create_sandbox_network()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise RuntimeError(f"Docker initialization failed: {str(e)}")
        
        self.running_containers = {}
        self.ports_mapping = {}
    
    def _create_sandbox_network(self):
        """Create an isolated network for sandboxes"""
        try:
            networks = self.client.networks.list(names=["kaleidoscope-sandbox"])
            if not networks:
                self.client.networks.create(
                    name="kaleidoscope-sandbox",
                    driver="bridge",
                    internal=False,  # Allow internet access but isolate containers
                    attachable=True,
                    options={
                        "com.docker.network.bridge.name": "kaleidoscope-sandbox",
                        "com.docker.network.bridge.enable_ip_masquerade": "true",
                        "com.docker.network.bridge.enable_icc": "false"  # Disable inter-container communication
                    }
                )
                logger.info("Created kaleidoscope-sandbox network")
            else:
                logger.info("Using existing kaleidoscope-sandbox network")
        except Exception as e:
            logger.error(f"Failed to create sandbox network: {str(e)}")
            # Continue anyway, we'll use default network if needed
    
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
        
        dockerfile = "# Generated by Kaleidoscope AI Sandbox\n"
        
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
            dockerfile += "COPY requirements.txt /app/requirements.txt 2>/dev/null || echo 'No requirements.txt found'\n"
            dockerfile += "RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n"
            
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
            elif framework == "fastapi":
                dockerfile += "EXPOSE 8000\n"
                dockerfile += 'CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]\n'
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
            elif framework == "angular":
                dockerfile += "FROM node:16-alpine\n\n"
                dockerfile += "WORKDIR /app\n\n"
                dockerfile += "COPY package*.json ./\n"
                dockerfile += "RUN npm install\n\n"
                dockerfile += "COPY . .\n\n"
                dockerfile += "EXPOSE 4200\n"
                dockerfile += 'CMD ["npm", "start", "--", "--host=0.0.0.0"]\n'
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
            
        elif language == "c" or language == "cpp":
            dockerfile += "FROM gcc:11\n\n"
            dockerfile += "WORKDIR /app\n\n"
            dockerfile += "COPY . .\n\n"
            # If it's a CMake project
            if os.path.exists(os.path.join(config.app_dir, "CMakeLists.txt")):
                dockerfile += "RUN apt-get update && apt-get install -y cmake\n"
                dockerfile += "RUN mkdir build && cd build && cmake .. && make\n\n"
                dockerfile += "EXPOSE 8080\n"
                dockerfile += 'CMD ["./build/app"]\n'
            else:
                # Simple compilation
                dockerfile += "RUN gcc -o app main.c\n\n"
                dockerfile += "EXPOSE 8080\n"
                dockerfile += 'CMD ["./app"]\n'
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
            # Check if we're at max containers
            if len(self.running_containers) >= SANDBOX_MAX_CONTAINERS:
                oldest_container = min(self.running_containers.items(), key=lambda x: x[1].attrs['Created'])
                logger.warning(f"Max containers reached, stopping oldest container: {oldest_container[0]}")
                self.cleanup_sandbox(oldest_container[0])
            
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
                'cpu_quota': int(100000 * config.cpu_limit),
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
                network="kaleidoscope-sandbox" if config.network_enabled else "none",
                read_only=False,  # Allow writing to container filesystem
                cap_drop=["ALL"],  # Drop all capabilities for security
                security_opt=["no-new-privileges:true"],  # Prevent privilege escalation
                restart_policy={"Name": "no"},  # Don't restart automatically
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
        import socket
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
                    import socket
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
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        """Initialize the sandbox manager"""
        self.docker_sandbox = DockerSandbox()
        self.db_pool = db_pool
        self.active_sandboxes = {}
    
    def initialize_db_pool(self, db_pool: asyncpg.Pool):
        """Initialize database pool"""
        self.db_pool = db_pool
    
    async def create_sandbox(self, sandbox_id: str, source_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a sandbox environment
        
        Args:
            sandbox_id: Sandbox ID
            source_path: Path to the source code
            options: Sandbox options
            
        Returns:
            Sandbox information
        """
        try:
            logger.info(f"Creating sandbox {sandbox_id} for {source_path}")
            
            # Generate unique container name
            safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '', options.get('app_name', 'app').lower())
            container_name = f"kaleidoscope-{safe_name}-{secrets.token_hex(4)}"
            
            # Determine app type and default ports
            app_type = options.get('app_type', 'web')
            language = options.get('language', 'python')
            framework = options.get('framework', 'generic')
            
            # Set default ports based on framework
            expose_ports = options.get('expose_ports', [])
            if not expose_ports:
                if framework == "flask":
                    expose_ports = [5000]
                elif framework == "django" or framework == "fastapi":
                    expose_ports = [8000]
                elif framework == "express" or framework == "react":
                    expose_ports = [3000]
                elif framework == "angular":
                    expose_ports = [4200]
                else:
                    expose_ports = [8080]
            
            # Create sandbox configuration
            config = SandboxConfig(
                sandbox_id=sandbox_id,
                app_dir=source_path,
                app_name=options.get('app_name', 'app'),
                app_type=app_type,
                language=language,
                framework=framework,
                memory_limit=options.get('memory_limit', DEFAULT_MEMORY_LIMIT),
                cpu_limit=options.get('cpu_limit', float(DEFAULT_CPU_LIMIT)),
                network_enabled=options.get('network_enabled', True),
                expose_ports=expose_ports,
                mount_volumes=options.get('mount_volumes', {}),
                environment=options.get('environment', {}),
                timeout_seconds=options.get('timeout_seconds', SANDBOX_TIMEOUT_SECONDS),
                container_name=container_name
            )
            
            # Run sandbox
            result = self.docker_sandbox.run_sandbox(config)
            
            # Store sandbox info in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                    UPDATE sandboxes 
                    SET container_id = $1, container_name = $2, status = $3, ports = $4
                    WHERE id = $5
                    """,
                    result["container_id"],
                    container_name,
                    result["status"],
                    json.dumps(result["ports"]),
                    sandbox_id
                    )
            
            # Store in memory
            self.active_sandboxes[sandbox_id] = {
                "container_id": result["container_id"],
                "container_name": container_name,
                "config": config,
                "result": result,
                "created_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {str(e)}")
            # Update database with error
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                    UPDATE sandboxes 
                    SET status = 'error', logs = $1
                    WHERE id = $2
                    """,
                    str(e),
                    sandbox_id
                    )
            raise
    
    def get_sandbox_logs(self, sandbox_id: str) -> str:
        """
        Get logs for a sandbox
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Sandbox logs
        """
        if sandbox_id not in self.active_sandboxes:
            return "Sandbox not found"
        
        container_name = self.active_sandboxes[sandbox_id]["container_name"]
        return self.docker_sandbox.get_container_logs(container_name)
    
    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """
        Terminate a sandbox
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            Success status
        """
        logger.info(f"Terminating sandbox {sandbox_id}")
        
        if sandbox_id not in self.active_sandboxes:
            logger.warning(f"Sandbox {sandbox_id} not found in active sandboxes")
            
            # Try to find in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    sandbox = await conn.fetchrow("""
                    SELECT container_name FROM sandboxes WHERE id = $1
                    """, sandbox_id)
                    
                    if sandbox and sandbox["container_name"]:
                        success = self.docker_sandbox.cleanup_sandbox(sandbox["container_name"])
                        
                        # Update database
                        await conn.execute("""
                        UPDATE sandboxes SET status = 'terminated' WHERE id = $1
                        """, sandbox_id)
                        
                        return success
            
            return False
        
        # Get container name
        container_name = self.active_sandboxes[sandbox_id]["container_name"]
        
        # Clean up sandbox
        success = self.docker_sandbox.cleanup_sandbox(container_name)
        
        # Update database
        if self.db_pool and success:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                UPDATE sandboxes SET status = 'terminated' WHERE id = $1
                """, sandbox_id)
        
        # Remove from active sandboxes
        if success:
            del self.active_sandboxes[sandbox_id]
        
        return success

    async def check_expired_sandboxes(self):
        """Check and terminate expired sandboxes"""
        logger.info("Checking for expired sandboxes")
        
        if not self.db_pool:
            logger.warning("Database pool not initialized, skipping expired sandbox check")
            return
        
        async with self.db_pool.acquire() as conn:
            # Find expired sandboxes
            expired = await conn.fetch("""
            SELECT id, container_name, container_id
            FROM sandboxes
            WHERE status IN ('running', 'starting') AND expires_at < NOW()
            """)
            
            logger.info(f"Found {len(expired)} expired sandboxes")
            
            # Terminate each expired sandbox
            for sandbox in expired:
                sandbox_id = sandbox["id"]
                container_name = sandbox["container_name"]
                
                logger.info(f"Terminating expired sandbox {sandbox_id} ({container_name})")
                
                if container_name:
                    success = self.docker_sandbox.cleanup_sandbox(container_name)
                    
                    # Update database
                    if success:
                        await conn.execute("""
                        UPDATE sandboxes SET status = 'expired' WHERE id = $1
                        """, sandbox_id)
                        
                        # Remove from active sandboxes
                        if sandbox_id in self.active_sandboxes:
                            del self.active_sandboxes[sandbox_id]
                else:
                    # Just mark as expired if no container name
                    await conn.execute("""
                    UPDATE sandboxes SET status = 'expired' WHERE id = $1
                    """, sandbox_id)

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database connection pool
    logger.info("Initializing database connection")
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    
    # Create and initialize sandbox manager
    sandbox_manager = SandboxManager(db_pool)
    app.state.db_pool = db_pool
    app.state.sandbox_manager = sandbox_manager
    
    # Start background tasks
    expired_check_task = None
    
    async def check_expired_sandboxes_task():
        while True:
            try:
                await sandbox_manager.check_expired_sandboxes()
            except Exception as e:
                logger.error(f"Error checking expired sandboxes: {str(e)}")
            
            # Check every minute
            await asyncio.sleep(60)
    
    expired_check_task = asyncio.create_task(check_expired_sandboxes_task())
    
    yield
    
    # Clean up
    logger.info("Shutting down sandbox manager")
    if expired_check_task:
        expired_check_task.cancel()
        try:
            await expired_check_task
        except asyncio.CancelledError:
            pass
    
    # Clean up all sandboxes
    sandbox_manager.docker_sandbox.cleanup_all()
    
    # Close database pool
    if app.state.db_pool:
        await app.state.db_pool.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_healthy = False
        try:
            async with app.state.db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                db_healthy = result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
        
        # Check Docker
        docker_healthy = False
        try:
            docker_client = docker.from_env()
            docker_client.ping()
            docker_healthy = True
        except Exception as e:
            logger.error(f"Docker health check failed: {str(e)}")
        
        # System health
        system_stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "active_sandboxes": len(app.state.sandbox_manager.active_sandboxes)
        }
        
        return {
            "status": "healthy" if db_healthy and docker_healthy else "degraded",
            "database": "healthy" if db_healthy else "unhealthy",
            "docker": "healthy" if docker_healthy else "unhealthy",
            "system": system_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/sandbox/{sandbox_id}/provision")
async def provision_sandbox(sandbox_id: str, options: Dict[str, Any]):
    """
    Provision a new sandbox
    
    Args:
        sandbox_id: Sandbox ID
        options: Sandbox options
    """
    try:
        # Validate required fields
        required_fields = ["source_path"]
        for field in required_fields:
            if field not in options:
                raise HTTPException(