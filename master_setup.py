#!/usr/bin/env python3
"""
Master reorganization script for Unravel AI project.
This script orchestrates the entire repository reorganization process.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = Path(os.getcwd()).resolve()
OUTPUT_DIR = ROOT_DIR / "unified_repo"
BACKUP_DIR = ROOT_DIR / ".backup"
TEMP_DIR = ROOT_DIR / ".temp"

def validate_environment():
    """Validate that we're in the correct repository environment."""
    markers = ["README.md", "requirements.txt", "src"]
    missing = [m for m in markers if not (ROOT_DIR / m).exists()]
    
    if missing:
        print(f"Warning: This doesn't appear to be the Unravel AI repository root.")
        print(f"Missing files/directories: {', '.join(missing)}")
        return False
    
    return True

def create_scripts():
    """Generate placeholder reorganization and analysis scripts."""
    scripts = {
        "reorganizer.py": "#!/usr/bin/env python3\nprint('Placeholder reorganizer script')",
        "dependency_analyzer.py": "#!/usr/bin/env python3\nprint('Placeholder dependency analyzer')",
        "dockerfile_generator.py": "#!/usr/bin/env python3\nprint('Placeholder Dockerfile generator')",
        "execution_script.py": "#!/usr/bin/env python3\nprint('Placeholder execution script')",
    }
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    for script_name, content in scripts.items():
        script_path = TEMP_DIR / script_name
        with open(script_path, "w") as f:
            f.write(content)
        script_path.chmod(script_path.stat().st_mode | 0o755)
    
    return list(scripts.keys())

def run_reorganization(args):
    """Run the repository reorganization process."""
    print("Starting repository reorganization...")
    
    cmd = [
        sys.executable,
        str(TEMP_DIR / "execution_script.py"),
        "--force" if args.force else "",
        "--backup" if args.backup else "",
        "--replace" if args.replace else "",
        "--skip-verification" if args.skip_verification else "",
        "--skip-analysis" if args.skip_analysis else "",
        "--keep-temp" if args.keep_temp else "",
    ]
    
    cmd = [arg for arg in cmd if arg]
    
    try:
        subprocess.run(cmd, check=True)
        print("Repository reorganization completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during reorganization: {e}")
        return False

def generate_dockerfiles(args):
    """Generate Dockerfiles for all services."""
    if not (OUTPUT_DIR / "src").exists():
        print("Error: Reorganized repository not found. Run reorganization first.")
        return False
    
    print("Generating Dockerfiles...")
    
    cmd = [
        sys.executable,
        str(TEMP_DIR / "dockerfile_generator.py"),
        "--root", str(OUTPUT_DIR),
        "--cuda" if args.cuda else "",
    ]
    
    cmd = [arg for arg in cmd if arg]
    
    try:
        subprocess.run(cmd, check=True)
        print("Dockerfile generation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during Dockerfile generation: {e}")
        return False

def clean_up(args):
    """Clean up temporary files."""
    if not args.keep_temp and TEMP_DIR.exists():
        print("Cleaning up temporary files...")
        shutil.rmtree(TEMP_DIR)

def create_deploy_script():
    """Create a deployment script for the reorganized repository."""
    deploy_script_path = OUTPUT_DIR / "deploy.sh"
    
    deploy_script_content = """#!/bin/bash
# Deployment script for Unravel AI
set -e

# Default environment
ENV=${1:-prod}

echo "Deploying Unravel AI to $ENV environment..."

# Build Docker images
docker-compose -f docker/docker-compose.yml build

# Start services in detached mode
docker-compose -f docker/docker-compose.yml up -d

echo "Deployment completed."
echo "API available at: http://localhost:8000"
echo "To stop all services: docker-compose -f docker/docker-compose.yml down"
"""
    
    with open(deploy_script_path, "w") as f:
        f.write(deploy_script_content)
    
    deploy_script_path.chmod(deploy_script_path.stat().st_mode | 0o755)
    
    print(f"Created deployment script at {deploy_script_path}")

def create_ci_workflow():
    """Create GitHub Actions CI workflow file."""
    github_dir = OUTPUT_DIR / ".github" / "workflows"
    github_dir.mkdir(parents=True, exist_ok=True)
    
    ci_workflow_path = github_dir / "ci.yml"
    
    ci_workflow_content = """name: Unravel AI CI

on:
  push:
    branches: [ main, master, dev ]
  pull_request:
    branches: [ main, master, dev ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt pytest
      - name: Run tests
        run: pytest tests/
"""
    
    with open(ci_workflow_path, "w") as f:
        f.write(ci_workflow_content)
    
    print(f"Created CI workflow at {ci_workflow_path}")

def generate_llm_integration():
    """Generate LLM integration module for the reorganized repository."""
    llm_dir = OUTPUT_DIR / "src" / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    llm_service_path = llm_dir / "service.py"
    llm_service_content = """#!/usr/bin/env python3
import fastapi
import uvicorn

app = fastapi.FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
"""
    
    with open(llm_service_path, "w") as f:
        f.write(llm_service_content)
    
    print(f"Generated LLM integration module at {llm_dir}")

def ensure_test_structure():
    """Ensure proper test directory structure exists."""
    test_dir = OUTPUT_DIR / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for module in ["api", "core", "db", "utils", "sandbox", "llm"]:
        module_test_dir = test_dir / module
        module_test_dir.mkdir(parents=True, exist_ok=True)
        (module_test_dir / "__init__.py").touch()
    
    print(f"Created test structure at {test_dir}")

def create_makefile():
    """Create a Makefile for common operations."""
    makefile_path = OUTPUT_DIR / "Makefile"
    
    makefile_content = """# Unravel AI Makefile

.PHONY: all install test clean

all: install test

install:
	pip install -r requirements.txt

test:
	pytest tests/

clean:
	rm -rf *.egg-info __pycache__ *.pyc
"""
    
    with open(makefile_path, "w") as f:
        f.write(makefile_content)
    
    print(f"Created Makefile at {makefile_path}")

def create_cli():
    """Create a CLI script for managing the Unravel AI project."""
    cli_path = OUTPUT_DIR / 'scripts' / 'cli.py'
    cli_dir = OUTPUT_DIR / 'scripts'
    cli_dir.mkdir(parents=True, exist_ok=True)
    
    cli_content = """#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def start_service(service, env="dev"):
    cmd = f"./launch.sh {env} {service}"
    try:
        run_command(cmd)
    except subprocess.CalledProcessError:
        print(f"Error starting {service} service")
        sys.exit(1)

def build_docker():
    cmd = "cd docker && docker-compose build"
    try:
        run_command(cmd)
    except subprocess.CalledProcessError:
        print("Error building Docker images")
        sys.exit(1)

def deploy():
    cmd = "./deploy.sh"
    try:
        run_command(cmd)
    except subprocess.CalledProcessError:
        print("Error deploying application")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unravel AI CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Service commands
    service_parser = subparsers.add_parser("service", help="Start services")
    service_parser.add_argument("name", choices=["api", "sandbox", "llm", "all"], help="Service to start")
    service_parser.add_argument("--env", "-e", choices=["dev", "prod"], default="dev", help="Environment")
    
    # Docker commands
    docker_parser = subparsers.add_parser("docker", help="Docker operations")
    docker_parser.add_argument("action", choices=["build", "up", "down"], help="Docker action")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy application")
    
    args = parser.parse_args()
    
    if args.command == "service":
        start_service(args.name, args.env)
    elif args.command == "docker":
        if args.action == "build":
            build_docker()
        elif args.action == "up":
            run_command("cd docker && docker-compose up -d")
        elif args.action == "down":
            run_command("cd docker && docker-compose down")
    elif args.command == "deploy":
        deploy()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
"""
    
    with open(cli_path, "w", encoding='utf-8') as f:
        f.write(cli_content)
    
    cli_path.chmod(0o755)
    print(f"Created CLI script at {cli_path}")

def create_readme():
    """Create a README.md with getting started instructions."""
    readme_path = OUTPUT_DIR / "README.md"
    readme_content = """# Unravel AI

## Getting Started

1. Run `./setup.sh` to set up the environment
2. Run `./launch.sh` to start the services

## Services

- API service: `./launch.sh dev api` (port 8000)
- Sandbox service: `./launch.sh dev sandbox` (port 8050)
- LLM service: `./launch.sh dev llm` (port 8100)

## Docker

Run with Docker: `cd docker && docker-compose up`

## Configuration

Example `config.json`:

```json
{
    "categorization": {
        "custom_rules": [
            {"pattern": ".*special.*", "category": "special"}
        ]
    },
    "dependencies": {
        "numpy": "1.26.4",
        "fastapi": "0.111.0"
    },
    "services": {
        "api": "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    }
}
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Created README at {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Unravel AI Master Repository Reorganizer")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing directories")
    parser.add_argument("--backup", action="store_true", help="Create backup of original repository")
    parser.add_argument("--replace", action="store_true", help="Replace original repository with reorganized version")
    parser.add_argument("--skip-verification", action="store_true", help="Skip repository verification step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip dependency analysis")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Dockerfile generation")
    parser.add_argument("--cuda", action="store_true", help="Generate CUDA-enabled Dockerfiles")
    parser.add_argument("--full", action="store_true", help="Perform full reorganization with all features")
    
    args = parser.parse_args()
    
    if args.full:
        args.backup = True
        args.cuda = True
    
    print("Unravel AI Repository Reorganizer")
    print("=================================")
    print(f"Repository: {ROOT_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    print()
    
    try:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        
        if not BACKUP_DIR.exists() and args.backup:
            BACKUP_DIR.mkdir(parents=True)
            for item in ROOT_DIR.glob('*'):
                if item.name.startswith('.') or item == BACKUP_DIR or item == OUTPUT_DIR:
                    continue
                shutil.copytree(item, BACKUP_DIR / item.name, dirs_exist_ok=True)
            print(f"Backup created at {BACKUP_DIR}")
        
        if not args.skip_verification and not validate_environment() and not args.force:
            print("Use --force to proceed anyway.")
            sys.exit(1)
        
        print("Creating reorganization scripts...")
        created_scripts = create_scripts()
        print(f"Created {len(created_scripts)} scripts in {TEMP_DIR}")
        
        if run_reorganization(args):
            if not args.skip_docker:
                generate_dockerfiles(args)
            
            create_deploy_script()
            create_ci_workflow()
            generate_llm_integration()
            ensure_test_structure()
            create_makefile()
            create_cli()
            create_readme()
            
            print("\nRepository reorganization completed successfully!")
            print(f"Reorganized repository is available at: {OUTPUT_DIR}")
            
            if args.replace:
                for item in OUTPUT_DIR.glob('*'):
                    shutil.move(str(item), str(ROOT_DIR / item.name))
                shutil.rmtree(OUTPUT_DIR)
                print("Original repository replaced with reorganized version.")
            
            print("\nNext steps:")
            print("1. cd unified_repo (or stay in current dir if --replace was used)")
            print("2. Run './setup.sh' to set up the environment")
            print("3. Run './launch.sh dev' to start all services")
            print("4. Use './scripts/cli.py --help' for CLI commands")
        
        clean_up(args)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        clean_up(args)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        clean_up(args)
        sys.exit(1)

if __name__ == "__main__":
    main()
