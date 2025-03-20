#!/usr/bin/env python3
"""
Kaleidoscope AI - Sandbox Execution Environment
===============================================
Provides a secure, isolated sandbox for executing generated applications.
Allows users to test apps immediately after generation with full networking,
database, and UI capabilities, then cleans up all resources automatically.
"""

import os
import sys
import time
import json
import logging
import asyncio
import shutil
import signal
import uuid
import tempfile
import subprocess
import socket
import threading
import webbrowser
import secrets
import docker
import tarfile
import io
import re
import platform
import psutil
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("sandbox.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
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
        if not self.container_name:
            # Generate a unique container name
            safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '', self.app_name.lower())
            self.container_name = f"kaleidoscope-{safe_name}-{secrets.token_hex(4)}"
        
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

class KubernetesAdapter:
    """Interface with Kubernetes for more advanced sandbox deployment"""

    def __init__(self, namespace: str = "kaleidoscope-sandbox"):
        """
        Initialize the Kubernetes adapter
        
        Args:
            namespace: Kubernetes namespace for sandboxes
        """
        try:
            from kubernetes import client, config
            
            # Try to load config from default locations
            try:
                config.load_kube_config()
            except Exception:
                # Fall back to in-cluster config
                config.load_incluster_config()
            
            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            self.namespace = namespace
            
            # Create namespace if it doesn't exist
            try:
                self.core_api.read_namespace(name=namespace)
            except Exception:
                ns = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.core_api.create_namespace(body=ns)
                logger.info(f"Created Kubernetes namespace: {namespace}")
            
            self.enabled = True
            logger.info("Kubernetes adapter initialized")
        except Exception as e:
            logger.warning(f"Kubernetes adapter not available: {str(e)}")
            self.enabled = False
    
    def deploy_app(self, config: SandboxConfig) -> Dict[str, Any]:
        """
        Deploy app to Kubernetes
        
        Args:
            config: Sandbox configuration
            
        Returns:
            Deployment information
        """
        if not self.enabled:
            raise RuntimeError("Kubernetes adapter not available")
        
        # Implementation details for Kubernetes deployment would go here
        # This would include creating deployments, services, ingress, etc.
        
        return {
            "status": "deployed",
            "namespace": self.namespace,
            "app_name": config.app_name
        }
    
    def cleanup_app(self, app_name: str) -> bool:
        """
        Clean up a deployed app
        
        Args:
            app_name: App name
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        # Implementation details for Kubernetes cleanup would go here
        
        return True

class SandboxManager:
    """Manages sandbox environments"""
    
    def __init__(self):
        """Initialize the sandbox manager"""
        self.docker_sandbox = DockerSandbox()
        self.k8s_adapter = KubernetesAdapter()
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
        
        # Choose deployment method based on configuration
        if app_config.get('use_kubernetes', False) and self.k8s_adapter.enabled:
            result = self.k8s_adapter.deploy_app(config)
        else:
            result = self.docker_sandbox.run_sandbox(config)
        
        # Store active sandbox info
        self.active_sandboxes[config.container_name] = {
            "config": config,
            "info": result,
            "start_time": time.time(),
            "deployment_type": "kubernetes" if app_config.get('use_kubernetes', False) else "docker"
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
        deployment_type = sandbox["deployment_type"]
        
        if deployment_type == "docker":
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
                "type": deployment_type,
                "start_time": sandbox["start_time"],
                "elapsed_time": time.time() - sandbox["start_time"],
                "ports": ports,
                "urls": urls
            }
        elif deployment_type == "kubernetes":
            # Implementation for Kubernetes status would go here
            return {
                "id": sandbox_id,
                "status": "running",  # Placeholder
                "type": deployment_type,
                "start_time": sandbox["start_time"],
                "elapsed_time": time.time() - sandbox["start_time"]
            }
        else:
            return {"status": "unknown_type"}
    
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
        
        sandbox = self.active_sandboxes[sandbox_id]
        deployment_type = sandbox["deployment_type"]
        
        if deployment_type == "docker":
            return self.docker_sandbox.get_container_logs(sandbox_id)
        elif deployment_type == "kubernetes":
            # Implementation for Kubernetes logs would go here
            return "Kubernetes logs not implemented"
        else:
            return "Unknown deployment type"
    
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
        
        sandbox = self.active_sandboxes[sandbox_id]
        deployment_type = sandbox["deployment_type"]
        
        # Clean up based on deployment type
        if deployment_type == "docker":
            success = self.docker_sandbox.cleanup_sandbox(sandbox_id)
        elif deployment_type == "kubernetes":
            success = self.k8s_adapter.cleanup_app(sandbox["config"].app_name)
        else:
            success = False
        
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

class ResourceMonitor:
    """Monitors system resources"""
    
    def __init__(self):
        """Initialize the resource monitor"""
        self.monitoring = False
        self.monitor_thread = None
        self.resource_data = []
        self.max_data_points = 60  # Store up to 60 data points
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            self.monitor_thread = None
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                disk = psutil.disk_usage('/')
                disk_