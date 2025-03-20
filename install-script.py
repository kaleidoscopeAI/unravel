#!/usr/bin/env python3
"""
Kaleidoscope AI System - Automated Setup and Launch Script
=========================================================
Handles complete setup, dependency installation, configuration, and launch
of the Kaleidoscope AI system for software analysis and mimicry.
"""

import os
import sys
import time
import json
import shutil
import platform
import subprocess
import argparse
import tempfile
import logging
import urllib.request
import zipfile
import tarfile
import signal
import re
import hashlib
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Configuration
# =============================================================================

SYSTEM_NAME = "Kaleidoscope AI"
SYSTEM_VERSION = "1.0.0"
SYSTEM_DESCRIPTION = "Software analysis and mimicry system"
DEFAULT_WORK_DIR = os.path.join(os.path.expanduser("~"), "kaleidoscope_workdir")
DEFAULT_PORT = 5050
REQUIRED_PYTHON_VERSION = (3, 8)  # Minimum Python version required

# External dependencies and their versions
DEPENDENCIES = {
    "python": [
        "aiohttp>=3.8.0",
        "backoff>=2.0.0",
        "cryptography>=38.0.0",
        "docker>=6.0.0",
        "flask>=2.0.0",
        "networkx>=2.8.0",
        "psutil>=5.9.0",
        "requests>=2.28.0",
        "tenacity>=8.1.0",
        "tiktoken>=0.3.0",
        "werkzeug>=2.2.0"
    ],
    "system": {
        "linux": ["build-essential", "python3-dev", "python3-pip", "docker.io"],
        "darwin": ["gcc", "docker"],
        "win32": ["Visual C++ Build Tools", "Docker Desktop"]
    },
    "optional": {
        "decompilers": {
            "ghidra": "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.1.5_build/ghidra_10.1.5_PUBLIC_20220726.zip",
            "retdec": "https://github.com/avast/retdec/releases/download/v5.0/retdec-v5.0-linux-64b.tar.xz"
        }
    }
}

# System modules to create/download
SYSTEM_MODULES = [
    "kaleidoscope_core.py",
    "system-upgrade-module.txt",
    "licensing-system.txt",
    "execution-sandbox-system.txt",
    "app-generator.txt",
    "code-reusability-module.txt",
    "error-handling-system.txt",
    "llm-integration-module.txt"
]

# User interfaces
UI_MODULES = [
    "kaleidoscope-chatbot.py",
    "kaleidoscope-web-interface.py"
]


# =============================================================================
# Configuration Manager
# =============================================================================

class SystemConfiguration:
    """
    Manages system configuration by reading/writing a JSON config file.
    """

    def __init__(self, config_path: str = "kaleidoscope_config.json"):
        """
        Initialize system configuration.

        Args:
            config_path (str): Path to configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading configuration: {str(e)}")

        # Default configuration
        return {
            "version": SYSTEM_VERSION,
            "work_dir": DEFAULT_WORK_DIR,
            "api_keys": {},
            "ui": {
                "web_enabled": True,
                "web_host": "127.0.0.1",
                "web_port": DEFAULT_PORT,
                "chatbot_enabled": True
            },
            "system": {
                "max_workers": multiprocessing.cpu_count(),
                "max_memory_gb": 4,
                "temp_dir": tempfile.gettempdir(),
                "decompilers": []
            },
            "modules": {
                "enabled": ["all"]
            }
        }

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key (str): Configuration key (dot notation supported).
            default (Any): Default value if key not found.

        Returns:
            Any: Configuration value.
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key (str): Configuration key (dot notation supported).
            value (Any): Value to set.
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the correct level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final value
        config[keys[-1]] = value


# =============================================================================
# System Installer
# =============================================================================

class SystemInstaller:
    """
    Handles installation of the Kaleidoscope system:
      - Verifies Python version
      - Installs Python/system dependencies
      - Sets up directories
      - Creates placeholder modules, UI scripts, etc.
    """

    def __init__(self, config: SystemConfiguration):
        """
        Initialize the installer.

        Args:
            config (SystemConfiguration): System configuration instance.
        """
        self.config = config
        self.system_dir = os.path.dirname(os.path.abspath(__file__))
        self.install_dir = os.path.join(self.system_dir, "kaleidoscope")
        self.modules_dir = os.path.join(self.install_dir, "modules")
        self.ui_dir = os.path.join(self.install_dir, "ui")
        self.templates_dir = os.path.join(self.ui_dir, "templates")
        self.static_dir = os.path.join(self.ui_dir, "static")

    def setup_directory_structure(self) -> None:
        """Create necessary directories for installation and runtime."""
        os.makedirs(self.install_dir, exist_ok=True)
        os.makedirs(self.modules_dir, exist_ok=True)
        os.makedirs(self.ui_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(os.path.join(self.static_dir, "css"), exist_ok=True)
        os.makedirs(os.path.join(self.static_dir, "js"), exist_ok=True)

        # Create working directory and subdirectories
        work_dir = self.config.get("work_dir")
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Created working directory: {work_dir}")

        for subdir in ["uploads", "decompiled", "specs", "reconstructed", "mimicked", "cache"]:
            os.makedirs(os.path.join(work_dir, subdir), exist_ok=True)

    def check_python_version(self) -> bool:
        """
        Check if the Python version is compatible.

        Returns:
            bool: True if Python is compatible, else False.
        """
        current_version = sys.version_info[:2]
        if current_version < REQUIRED_PYTHON_VERSION:
            logger.error(
                f"Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} "
                f"or higher is required (found {sys.version.split()[0]})"
            )
            return False

        logger.info(f"Python version {sys.version.split()[0]} is compatible")
        return True

    def install_python_dependencies(self) -> bool:
        """
        Install Python dependencies using pip.

        Returns:
            bool: True if successful, else False.
        """
        logger.info("Installing Python dependencies...")

        requirements_path = os.path.join(self.system_dir, "requirements.txt")
        try:
            with open(requirements_path, 'w') as f:
                f.write("\n".join(DEPENDENCIES["python"]))
        except Exception as e:
            logger.error(f"Failed to write requirements.txt: {str(e)}")
            return False

        # Install dependencies using pip
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            logger.info("Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing Python dependencies: {str(e)}")
            return False

    def install_system_dependencies(self) -> bool:
        """
        Install system-level dependencies based on the current platform.

        Returns:
            bool: True if successful (or not required), else False.
        """
        system = platform.system().lower()
        if system == "linux":
            return self._install_linux_dependencies()
        elif system == "darwin":
            return self._install_macos_dependencies()
        elif system == "windows":
            return self._check_windows_dependencies()
        else:
            logger.error(f"Unsupported operating system: {platform.system()}")
            return False

    def _install_linux_dependencies(self) -> bool:
        """
        Install Linux dependencies (best-effort) using detected package managers.

        Returns:
            bool: True if at least one recognized manager was found and installed, else False.
        """
        logger.info("Installing Linux dependencies (best-effort)...")
        packages = DEPENDENCIES["system"]["linux"]
        tried_install = False

        try:
            if os.path.exists("/usr/bin/apt-get"):
                tried_install = True
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y"] + packages)
            elif os.path.exists("/usr/bin/dnf"):
                tried_install = True
                subprocess.check_call(["sudo", "dnf", "install", "-y"] + packages)
            elif os.path.exists("/usr/bin/yum"):
                tried_install = True
                subprocess.check_call(["sudo", "yum", "install", "-y"] + packages)
            elif os.path.exists("/usr/bin/pacman"):
                tried_install = True
                subprocess.check_call(["sudo", "pacman", "-Sy", "--noconfirm"] + packages)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing Linux dependencies: {str(e)}")
            return False

        if not tried_install:
            logger.warning("No recognized package manager found. System dependencies not installed.")
            return False

        logger.info("Linux dependencies installed successfully (or no recognized manager used).")
        return True

    def _install_macos_dependencies(self) -> bool:
        """
        Install macOS dependencies via Homebrew (best-effort).

        Returns:
            bool: True if successful, else False.
        """
        logger.info("Installing macOS dependencies (best-effort)...")
        packages = DEPENDENCIES["system"]["darwin"]

        try:
            # Check if Homebrew is installed
            try:
                subprocess.check_call(["brew", "--version"], stdout=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Homebrew not found. Attempting to install...")
                brew_install_cmd = (
                    '/bin/bash -c "$(curl -fsSL '
                    'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                )
                subprocess.check_call(brew_install_cmd, shell=True)

            subprocess.check_call(["brew", "install"] + packages)
            logger.info("macOS dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing macOS dependencies: {str(e)}")
            return False

    def _check_windows_dependencies(self) -> bool:
        """
        Check for Windows dependencies: Docker Desktop & Visual C++ Build Tools.
        Prints instructions if missing.

        Returns:
            bool: True if dependencies are found, else False.
        """
        logger.info("Checking Windows dependencies (Docker Desktop, VC++ Build Tools)...")
        docker_installed = False
        vc_installed = False

        # Check Docker
        try:
            subprocess.check_call(["docker", "--version"], stdout=subprocess.DEVNULL)
            docker_installed = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Docker Desktop is not installed or not in PATH.")

        # Check Visual C++ Build Tools via `cl.exe`
        try:
            result = subprocess.check_output(["where", "cl.exe"], stderr=subprocess.STDOUT, universal_newlines=True)
            if "cl.exe" in result:
                vc_installed = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Visual C++ Build Tools are not installed or not in PATH.")

        if not (docker_installed and vc_installed):
            logger.info("Please ensure the following are installed on Windows:")
            if not docker_installed:
                logger.info(" - Docker Desktop: https://www.docker.com/products/docker-desktop")
            if not vc_installed:
                logger.info(" - Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            return False

        logger.info("Windows dependencies appear to be installed.")
        return True

    def setup_kaleidoscope_core(self) -> None:
        """
        Create or overwrite the Kaleidoscope core module with a placeholder
        that demonstrates ingestion, decompilation, specification, and mimicry.
        """
        core_path = os.path.join(self.install_dir, "kaleidoscope_core.py")
        logger.info("Creating Kaleidoscope core module...")

        core_content = r'''#!/usr/bin/env python3
"""
Kaleidoscope AI Core
====================
Core implementation of the Kaleidoscope AI system for software analysis and mimicry.
"""

import os
import sys
import time
import json
import logging
import subprocess
import tempfile
import shutil
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_core.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KaleidoscopeCore:
    """Core implementation of the Kaleidoscope AI system."""

    def __init__(self, work_dir: str = None):
        """
        Initialize the Kaleidoscope core.

        Args:
            work_dir (str): Working directory for Kaleidoscope.
        """
        self.work_dir = work_dir or os.path.join(os.getcwd(), "kaleidoscope_workdir")
        os.makedirs(self.work_dir, exist_ok=True)

        # Create subdirectories
        self.uploads_dir = os.path.join(self.work_dir, "uploads")
        self.decompiled_dir = os.path.join(self.work_dir, "decompiled")
        self.specs_dir = os.path.join(self.work_dir, "specs")
        self.reconstructed_dir = os.path.join(self.work_dir, "reconstructed")
        self.mimicked_dir = os.path.join(self.work_dir, "mimicked")

        for d in [self.uploads_dir, self.decompiled_dir, self.specs_dir,
                  self.reconstructed_dir, self.mimicked_dir]:
            os.makedirs(d, exist_ok=True)

        logger.info(f"Initialized Kaleidoscope core in {self.work_dir}")

    def ingest_software(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest software for analysis.

        Args:
            file_path (str): Path to the software file.

        Returns:
            Dict[str, Any]: Result dictionary.
        """
        logger.info(f"Ingesting software: {file_path}")

        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}"
            }

        ingestion_id = str(uuid.uuid4())

        # Create ingestion-specific directories
        decompiled_dir = os.path.join(self.decompiled_dir, ingestion_id)
        specs_dir = os.path.join(self.specs_dir, ingestion_id)
        reconstructed_dir = os.path.join(self.reconstructed_dir, ingestion_id)

        for d in [decompiled_dir, specs_dir, reconstructed_dir]:
            os.makedirs(d, exist_ok=True)

        try:
            # Decompile software (placeholder)
            decompiled_files = self._decompile_software(file_path, decompiled_dir)
            # Generate specifications
            spec_files = self._generate_specifications(decompiled_files, specs_dir)
            # Reconstruct software
            reconstructed_files = self._reconstruct_software(spec_files, reconstructed_dir)

            return {
                "status": "completed",
                "ingestion_id": ingestion_id,
                "file_path": file_path,
                "decompiled_files": decompiled_files,
                "spec_files": spec_files,
                "reconstructed_files": reconstructed_files
            }
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _decompile_software(self, file_path: str, output_dir: str) -> List[str]:
        """
        Decompile software (placeholder).

        Args:
            file_path (str): Path to the software file.
            output_dir (str): Output directory for decompiled results.

        Returns:
            List[str]: Paths to decompiled files.
        """
        logger.info(f"Decompiling {file_path}...")
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.cs']:
            output_file = os.path.join(output_dir, os.path.basename(file_path))
            shutil.copy(file_path, output_file)
            return [output_file]
        elif ext in ['.exe', '.dll', '.so', '.dylib']:
            output_file = os.path.join(output_dir, os.path.basename(file_path) + ".decompiled.txt")
            with open(output_file, 'w') as f:
                f.write(f"// Decompiled from {file_path}\n")
                f.write("// This is a placeholder for actual decompilation output\n")
            return [output_file]
        else:
            output_file = os.path.join(output_dir, os.path.basename(file_path) + ".txt")
            with open(output_file, 'w') as f:
                f.write(f"// Analysis of {file_path}\n")
                f.write("// This is a placeholder for actual analysis output\n")
            return [output_file]

    def _generate_specifications(self, decompiled_files: List[str], output_dir: str) -> List[str]:
        """
        Generate specifications from decompiled files (placeholder).

        Args:
            decompiled_files (List[str]): List of decompiled file paths.
            output_dir (str): Output directory for spec files.

        Returns:
            List[str]: List of specification file paths.
        """
        logger.info("Generating specifications...")
        spec_files = []

        for file_path in decompiled_files:
            spec_file = os.path.join(output_dir, os.path.basename(file_path) + ".spec.md")
            with open(spec_file, 'w') as f_out:
                f_out.write(f"# Specification for {os.path.basename(file_path)}\n\n")
                f_out.write("## Overview\n\nPlaceholder for software specs.\n")
            spec_files.append(spec_file)

        return spec_files

    def _reconstruct_software(self, spec_files: List[str], output_dir: str) -> List[str]:
        """
        Reconstruct software from specifications (placeholder).

        Args:
            spec_files (List[str]): Paths to specification files.
            output_dir (str): Output directory for reconstructed code.

        Returns:
            List[str]: Paths to reconstructed files.
        """
        logger.info("Reconstructing software...")
        reconstructed_files = []

        for spec_file in spec_files:
            base_name = os.path.basename(spec_file).replace(".spec.md", "")
            reconstructed_file = os.path.join(output_dir, base_name)
            with open(reconstructed_file, 'w') as f_out:
                f_out.write(f"// Reconstructed from {os.path.basename(spec_file)}\n")
                f_out.write("// Placeholder for actual reconstructed code\n")
            reconstructed_files.append(reconstructed_file)

        return reconstructed_files

    def mimic_software(self, spec_files: List[str], target_language: str) -> Dict[str, Any]:
        """
        Mimic software in a different language (placeholder).

        Args:
            spec_files (List[str]): List of specification file paths.
            target_language (str): Target programming language.

        Returns:
            Dict[str, Any]: Result dictionary.
        """
        logger.info(f"Mimicking software in {target_language}...")

        for spec_file in spec_files:
            if not os.path.exists(spec_file):
                return {
                    "status": "error",
                    "error": f"Spec file not found: {spec_file}"
                }

        mimicry_id = str(uuid.uuid4())
        mimic_dir = os.path.join(self.mimicked_dir, f"{mimicry_id}_{target_language}")
        os.makedirs(mimic_dir, exist_ok=True)

        try:
            mimicked_files = self._generate_mimicked_code(spec_files, target_language, mimic_dir)
            return {
                "status": "completed",
                "mimicry_id": mimicry_id,
                "target_language": target_language,
                "mimicked_files": mimicked_files
            }
        except Exception as e:
            logger.error(f"Error during mimicry: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_mimicked_code(self, spec_files: List[str], target_language: str, output_dir: str) -> List[str]:
        """
        Generate code in a target language from specs (placeholder).

        Args:
            spec_files (List[str]): Specification file paths.
            target_language (str): Language to generate.
            output_dir (str): Output directory.

        Returns:
            List[str]: Paths to generated code files.
        """
        logger.info(f"Generating {target_language} code (placeholder)...")
        extensions = {
            "python": ".py", "javascript": ".js", "java": ".java", 
            "c": ".c", "cpp": ".cpp", "c++": ".cpp", "csharp": ".cs", 
            "go": ".go", "ruby": ".rb", "rust": ".rs", "swift": ".swift"
        }
        ext = extensions.get(target_language.lower(), ".txt")
        mimicked_files = []

        for spec_file in spec_files:
            base_name = os.path.basename(spec_file).replace(".spec.md", "").split(".")[0]
            output_file = os.path.join(output_dir, f"{base_name}{ext}")
            with open(output_file, 'w') as f:
                f.write(f"// Mimicked from {spec_file}\n")
                f.write(f"// Placeholder {target_language} code\n")
            mimicked_files.append(output_file)

        return mimicked_files
'''

        with open(core_path, 'w') as f:
            f.write(core_content)

        logger.info(f"Created Kaleidoscope core module: {core_path}")

    def setup_system_modules(self) -> None:
        """
        Create placeholder system modules in the modules directory.
        """
        logger.info("Setting up system modules...")
        for module_file in SYSTEM_MODULES:
            module_path = os.path.join(self.modules_dir, module_file)
            with open(module_path, 'w') as f:
                f.write(f"""#!/usr/bin/env python3
'''
Kaleidoscope AI - {module_file.replace('-', ' ').replace('.txt', '').replace('.py', '').title()}
=========================================================
This is a placeholder for the actual module implementation.
'''

# Module implementation would go here
""")
            logger.info(f"Created module: {module_file}")

    def setup_ui_modules(self) -> None:
        """
        Create placeholder UI modules in the UI directory.
        """
        logger.info("Setting up UI modules...")
        for ui_file in UI_MODULES:
            ui_path = os.path.join(self.ui_dir, ui_file)
            title_name = ui_file.replace('-', ' ').replace('.py', '').title()
            with open(ui_path, 'w') as f:
                f.write(f"""#!/usr/bin/env python3
'''
Kaleidoscope AI - {title_name}
=========================================================
This is a placeholder for the actual UI implementation.
'''

import os
import sys
import logging
import argparse
from pathlib import Path

# Ensure the kaleidoscope_core module is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaleidoscope_core import KaleidoscopeCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    '''Main entry point for {title_name}'''
    parser = argparse.ArgumentParser(description="Kaleidoscope AI - {title_name}")
    parser.add_argument("--work-dir", "-w", help="Working directory", default=None)
    parser.add_argument("--port", help="Port for web interface", default=None)

    args = parser.parse_args()

    try:
        # Initialize Kaleidoscope core
        kaleidoscope = KaleidoscopeCore(work_dir=args.work_dir)

        # Start the UI (placeholder)
        logger.info("Starting {title_name}...")
        logger.info("{title_name} running (placeholder)")

    except Exception as e:
        logger.error(f"Error: {{str(e)}}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
            logger.info(f"Created UI module: {ui_file}")

    def create_main_script(self) -> None:
        """
        Create the main script that launches Kaleidoscope's web interface
        and/or chatbot in separate processes.
        """
        main_path = os.path.join(self.install_dir, "kaleidoscope.py")
        logger.info(f"Creating main script: {main_path}")

        main_script_content = r'''#!/usr/bin/env python3
"""
Kaleidoscope AI - Main Script
=============================
Main entry point for the Kaleidoscope AI system.
"""

import os
import sys
import logging
import argparse
import subprocess
import threading
import webbrowser
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KaleidoscopeLauncher:
    """Launcher for Kaleidoscope components."""

    def __init__(self, work_dir=None, web_port=5050, open_browser=True):
        """
        Initialize the launcher.

        Args:
            work_dir (str): Working directory for data.
            web_port (int): Port for the web interface.
            open_browser (bool): Whether to open a browser automatically.
        """
        self.install_dir = os.path.dirname(os.path.abspath(__file__))
        self.ui_dir = os.path.join(self.install_dir, "ui")
        self.work_dir = work_dir or os.path.join(os.path.expanduser("~"), "kaleidoscope_workdir")
        self.web_port = web_port
        self.open_browser = open_browser
        self.processes = []

    def launch_web_interface(self):
        """Launch the web interface in a subprocess."""
        web_script = os.path.join(self.ui_dir, "kaleidoscope-web-interface.py")
        if not os.path.exists(web_script):
            logger.error(f"Web interface script not found: {web_script}")
            return False

        logger.info(f"Launching web interface on port {self.web_port}...")
        try:
            process = subprocess.Popen(
                [sys.executable, web_script, "--work-dir", self.work_dir, "--port", str(self.web_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)

            # Wait a bit, then open browser
            time.sleep(2)
            if self.open_browser:
                web_url = f"http://localhost:{self.web_port}"
                webbrowser.open(web_url)
                logger.info(f"Opened web browser to {web_url}")

            return True
        except Exception as e:
            logger.error(f"Error launching web interface: {str(e)}")
            return False

    def launch_chatbot(self):
        """Launch the chatbot interface in a subprocess."""
        chatbot_script = os.path.join(self.ui_dir, "kaleidoscope-chatbot.py")
        if not os.path.exists(chatbot_script):
            logger.error(f"Chatbot script not found: {chatbot_script}")
            return False

        logger.info("Launching chatbot interface...")
        try:
            process = subprocess.Popen(
                [sys.executable, chatbot_script, "--work-dir", self.work_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)
            return True
        except Exception as e:
            logger.error(f"Error launching chatbot: {str(e)}")
            return False

    def monitor_output(self, process, name):
        """
        Monitor and log output from a process in real-time.

        Args:
            process (subprocess.Popen): The process to monitor.
            name (str): A label for logging output.
        """
        for line in iter(process.stdout.readline, ""):
            if line:
                logger.info(f"[{name}] {line.strip()}")

        for line in iter(process.stderr.readline, ""):
            if line:
                logger.error(f"[{name}] {line.strip()}")

    def wait_for_exit(self):
        """Wait until all processes have exited or Ctrl+C is received."""
        try:
            while any(p.poll() is None for p in self.processes):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping all processes...")
            self.stop_all()

    def stop_all(self):
        """Stop all running processes."""
        for p in self.processes:
            if p.poll() is None:
                p.terminate()

        time.sleep(1)
        for p in self.processes:
            if p.poll() is None:
                p.kill()

def main():
    """Main entry point for the Kaleidoscope system."""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI Launcher")
    parser.add_argument("--work-dir", "-w", help="Working directory", default=None)
    parser.add_argument("--port", "-p", type=int, help="Web interface port", default=5050)
    parser.add_argument("--web", action="store_true", help="Launch web interface only")
    parser.add_argument("--chatbot", action="store_true", help="Launch chatbot interface only")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # If neither or both flags are set, treat as launching both by default
    launch_web = args.web or (not args.web and not args.chatbot)
    launch_chatbot = args.chatbot or (not args.web and not args.chatbot)

    launcher = KaleidoscopeLauncher(
        work_dir=args.work_dir,
        web_port=args.port,
        open_browser=not args.no_browser
    )

    if launch_web:
        if not launcher.launch_web_interface():
            logger.error("Failed to launch web interface")

    if launch_chatbot:
        if not launcher.launch_chatbot():
            logger.error("Failed to launch chatbot interface")

    # Start threads to monitor subprocess output
    for i, process in enumerate(launcher.processes):
        name = f"Process-{i+1}"
        threading.Thread(target=launcher.monitor_output, args=(process, name), daemon=True).start()

    # Wait for processes or interrupt
    launcher.wait_for_exit()
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        with open(main_path, 'w') as f:
            f.write(main_script_content)

        os.chmod(main_path, 0o755)
        logger.info(f"Created main script: {main_path}")

    def run_full_installation(self) -> bool:
        """
        Run the full installation process.

        Returns:
            bool: True if everything installed successfully, else False.
        """
        logger.info(f"Starting {SYSTEM_NAME} installation...")

        try:
            # 1) Check Python version
            if not self.check_python_version():
                return False

            # 2) Set up directory structure
            self.setup_directory_structure()

            # 3) Install Python dependencies
            py_ok = self.install_python_dependencies()
            # 4) Install system dependencies
            sys_ok = self.install_system_dependencies()

            # Logging for partial successes
            if not py_ok:
                logger.warning("Some Python dependencies failed to install.")
            if not sys_ok:
                logger.warning("Some system dependencies are missing or failed to install.")

            # 5) Create or update the system files
            self.setup_kaleidoscope_core()
            self.setup_system_modules()
            self.setup_ui_modules()
            self.create_main_script()

            # 6) Save updated config
            self.config.save_config()

            logger.info(f"{SYSTEM_NAME} installation steps completed.")
            return True

        except Exception as e:
            logger.error(f"Installation failed: {str(e)}")
            return False


# =============================================================================
# Secondary Launcher Class
# =============================================================================

class KaleidoscopeAppLauncher:
    """
    Provides a programmatic API for launching the Kaleidoscope system
    outside the one-shot CLI setup.
    """

    def __init__(self, install_dir: str, work_dir: str = None, web_port: int = DEFAULT_PORT):
        """
        Initialize the launcher.

        Args:
            install_dir (str): Directory containing the kaleidoscope.py main script.
            work_dir (str): Working directory for the system.
            web_port (int): Port for the web UI.
        """
        self.install_dir = install_dir
        self.work_dir = work_dir or os.path.join(os.path.expanduser("~"), "kaleidoscope_workdir")
        self.web_port = web_port

        if self.install_dir not in sys.path:
            sys.path.insert(0, self.install_dir)

    def launch(self, interface: str = "web", open_browser: bool = True) -> int:
        """
        Launch the system.

        Args:
            interface (str): "web", "chatbot", or "both".
            open_browser (bool): Whether to open the browser for web UI.

        Returns:
            int: Exit code from the process call.
        """
        main_script = os.path.join(self.install_dir, "kaleidoscope.py")
        if not os.path.exists(main_script):
            logger.error(f"Main script not found: {main_script}")
            return 1

        cmd = [sys.executable, main_script, "--work-dir", self.work_dir, "--port", str(self.web_port)]
        if interface == "web":
            cmd.append("--web")
        elif interface == "chatbot":
            cmd.append("--chatbot")
        if not open_browser:
            cmd.append("--no-browser")

        try:
            logger.info(f"Launching {SYSTEM_NAME} with command: {' '.join(cmd)}")
            return subprocess.call(cmd)
        except Exception as e:
            logger.error(f"Error launching system: {str(e)}")
            return 1


# =============================================================================
# Entry Point
# =============================================================================

def run_setup():
    """
    Command-line entry point that:
      1) Parses args
      2) Installs Kaleidoscope
      3) Optionally launches the system
    """
    parser = argparse.ArgumentParser(description=f"{SYSTEM_NAME} Setup")
    parser.add_argument("--work-dir", "-w", help="Working directory", default=DEFAULT_WORK_DIR)
    parser.add_argument("--port", "-p", type=int, help="Web interface port", default=DEFAULT_PORT)
    parser.add_argument("--interface", "-i", choices=["web", "chatbot", "both"],
                        default="both", help="Interface to launch after installation")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--install-only", action="store_true", help="Only install; don't launch anything")

    args = parser.parse_args()

    # Load or create system configuration
    config = SystemConfiguration()
    config.set("work_dir", args.work_dir)
    config.set("ui.web_port", args.port)

    # Perform installation
    installer = SystemInstaller(config)
    installation_success = installer.run_full_installation()
    if not installation_success:
        logger.error("Installation encountered errors.")
        return 1

    # Launch if requested
    if not args.install_only:
        launcher = KaleidoscopeAppLauncher(
            install_dir=installer.install_dir,
            work_dir=args.work_dir,
            web_port=args.port
        )
        if args.interface == "both":
            return launcher.launch(interface="both", open_browser=not args.no_browser)
        else:
            return launcher.launch(interface=args.interface, open_browser=not args.no_browser)

    return 0


if __name__ == "__main__":
    sys.exit(run_setup())

