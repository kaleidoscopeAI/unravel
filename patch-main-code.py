#!/usr/bin/env python3
"""
Unravel AI - Patch for LLM Integration
Automatically applies patches to integrate LLM into Unravel AI
"""

import os
import sys
import re
import shutil
import argparse
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        "huggingface_hub",
        "requests",
        "psutil"
    ]
    
    missing = []
    for package in required:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Installing missing dependencies...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Dependencies installed successfully")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False
    
    return True

def patch_file(file_path, patterns):
    """
    Patch a file with multiple patterns
    
    Args:
        file_path: Path to the file to patch
        patterns: List of (pattern, replacement) tuples
    
    Returns:
        Whether the file was successfully patched
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if content != original_content:
        # Create backup
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        
        # Write patched file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Patched {file_path} (backup: {backup_path})")
        return True
    else:
        print(f"File already patched or patterns not found: {file_path}")
        return False

def patch_ingestion_py(unravel_dir):
    """Patch ingestion.py to integrate with LLM"""
    ingestion_py = os.path.join(unravel_dir, "src", "core", "ingestion.py")
    if not os.path.exists(ingestion_py):
        ingestion_py = os.path.join(unravel_dir, "ingestion.py")
    
    if not os.path.exists(ingestion_py):
        print(f"ingestion.py not found in {unravel_dir}")
        return False
    
    patterns = [
        # Add LLM import
        (
            r'(import tiktoken.*?)\n',
            r'\1\nimport importlib.util\n'
        ),
        # Modify SpecGenerator initialization
        (
            r'(class SpecGenerator:.*?def __init__\(self, work_dir: str = None\):.*?)(\s+self\.work_dir = work_dir or config\.SPECS_DIR)',
            r'\1\2\n        # Check if LLM module is available\n        self.use_llm = False\n        llm_spec = importlib.util.find_spec("src.core.llm_interface")\n        if llm_spec is not None:\n            self.use_llm = True\n            logger.info("LLM module detected, will use for enhanced specification generation")'
        ),
        # Modify generate_specifications to use LLM when available
        (
            r'(def generate_specifications\(self, decompiled_files: List\[str\]\) -> List\[str\]:.*?)(\s+# Ensure output directory exists)',
            r'\1\n        # Use LLM if available\n        if self.use_llm:\n            try:\n                from src.core.llm_interface import UnravelAIAnalyzer\n                \n                # Initialize analyzer\n                analyzer = UnravelAIAnalyzer()\n                \n                # Set up analyzer\n                if analyzer.setup():\n                    logger.info("Using LLM for specification generation")\n                    results = analyzer.analyze_software(decompiled_files, analysis_type="specification")\n                    \n                    # Process results\n                    for file_name, result in results.items():\n                        if file_name == "_summary":\n                            # Create combined specification\n                            spec_path = os.path.join(self.specs_dir, "combined_spec.md")\n                            with open(spec_path, "w") as f:\n                                f.write("# Software Specification\\n\\n")\n                                f.write("This document contains specifications extracted from the decompiled software.\\n\\n")\n                                f.write(result)\n                            spec_files.append(spec_path)\n                        else:\n                            # Find original file path\n                            file_path = next((f for f in decompiled_files if os.path.basename(f) == file_name), file_name)\n                            \n                            # Create file specification\n                            spec_path = os.path.join(self.specs_dir, f"{os.path.basename(file_path)}.spec.md")\n                            with open(spec_path, "w") as f:\n                                f.write(f"# {os.path.basename(file_path)} Specification\\n\\n")\n                                f.write(result)\n                            spec_files.append(spec_path)\n                    \n                    # Generate specialized specifications\n                    self._generate_specialized_specs_llm(analyzer, decompiled_files, spec_files)\n                    \n                    # Clean up\n                    analyzer.cleanup()\n                    \n                    return spec_files\n                else:\n                    logger.warning("Failed to set up LLM analyzer, falling back to standard specification generation")\n            except Exception as e:\n                logger.error(f"Error using LLM for specification generation: {str(e)}")\n                logger.warning("Falling back to standard specification generation")\n\2'
        ),
        # Add specialized specs generation method
        (
            r'(def _generate_specialized_specs\(.*?spec_files\).*?\n)([ ]{4})',
            r'\1\2\n\2def _generate_specialized_specs_llm(self, analyzer, decompiled_files, spec_files):\n\2    """Generate specialized specifications using LLM"""\n\2    try:\n\2        # Data structures spec\n\2        data_structures_path = os.path.join(self.specs_dir, "data_structures.md")\n\2        results = analyzer.analyze_software(decompiled_files, analysis_type="custom:Identify and document all data structures (classes, structs, enums, etc.) in these files. For each, include: name, fields with types, methods, relationships, and purpose.")\n\2        \n\2        with open(data_structures_path, "w") as f:\n\2            f.write("# Data Structures\\n\\n")\n\2            f.write("This document describes the key data structures identified in the software.\\n\\n")\n\2            if "_summary" in results:\n\2                f.write(results["_summary"])\n\2            else:\n\2                for file_name, result in results.items():\n\2                    if file_name != "error":\n\2                        f.write(f"## Data Structures in {file_name}\\n\\n")\n\2                        f.write(result)\n\2        \n\2        spec_files.append(data_structures_path)\n\2        \n\2        # API documentation\n\2        api_path = os.path.join(self.specs_dir, "api_documentation.md")\n\2        results = analyzer.analyze_software(decompiled_files, analysis_type="custom:Create comprehensive API documentation for these files. Document all public functions, methods, classes, and interfaces with their parameters, return values, and purpose.")\n\2        \n\2        with open(api_path, "w") as f:\n\2            f.write("# API Documentation\\n\\n")\n\2            f.write("This document describes the public API of the software.\\n\\n")\n\2            if "_summary" in results:\n\2                f.write(results["_summary"])\n\2            else:\n\2                for file_name, result in results.items():\n\2                    if file_name != "error":\n\2                        f.write(f"## API in {file_name}\\n\\n")\n\2                        f.write(result)\n\2        \n\2        spec_files.append(api_path)\n\2        \n\2        # Algorithm analysis\n\2        algorithm_path = os.path.join(self.specs_dir, "algorithms.md")\n\2        results = analyzer.analyze_software(decompiled_files, analysis_type="algorithm_detection")\n\2        \n\2        with open(algorithm_path, "w") as f:\n\2            f.write("# Algorithms\\n\\n")\n\2            f.write("This document describes the key algorithms identified in the software.\\n\\n")\n\2            if "_summary" in results:\n\2                f.write(results["_summary"])\n\2            else:\n\2                for file_name, result in results.items():\n\2                    if file_name != "error":\n\2                        f.write(f"## Algorithms in {file_name}\\n\\n")\n\2                        f.write(result)\n\2        \n\2        spec_files.append(algorithm_path)\n\2    \n\2    except Exception as e:\n\2        logger.error(f"Error generating specialized specs: {str(e)}")\n\2\n\2'
        )
    ]
    
    return patch_file(ingestion_py, patterns)

def patch_config_py(unravel_dir):
    """Patch config.py to add LLM settings"""
    config_py = os.path.join(unravel_dir, "config.py")
    
    if not os.path.exists(config_py):
        print(f"config.py not found in {unravel_dir}")
        return False
    
    patterns = [
        # Add LLM settings
        (
            r'([ ]{4}# Paths for decompilers.*?MIMICRY_DIR = os.path.join\(WORK_DIR, "mimicry"\))',
            r'\1\n\n    # LLM settings\n    LLM_CONFIG_PATH = os.path.expanduser("~/.config/unravel-ai/llm_config.json")\n    LLM_CACHE_DIR = os.path.expanduser("~/.cache/unravel-ai/models")\n    LLM_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"\n    LLM_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"\n    LLM_PROVIDER = "llamacpp_api"'
        )
    ]
    
    return patch_file(config_py, patterns)

def patch_setup_sh(unravel_dir):
    """Patch setup.sh to add LLM setup"""
    setup_sh = os.path.join(unravel_dir, "setup.sh")
    
    if not os.path.exists(setup_sh):
        print(f"setup.sh not found in {unravel_dir}")
        return False
    
    patterns = [
        # Add LLM setup
        (
            r'(echo "Setup complete!")',
            r'# Setup LLM module\necho "Setting up LLM module..."\nbash scripts/setup_llm.sh --skip-deps\n\n\1'
        )
    ]
    
    return patch_file(setup_sh, patterns)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Patch Unravel AI for LLM Integration")
    parser.add_argument("--unravel-dir", help="Path to Unravel AI directory")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine Unravel AI directory
    unravel_dir = args.unravel_dir
    if not unravel_dir:
        # Try to find in common locations
        candidates = [
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))),  # ./unravel-ai
            os.path.join(os.path.expanduser("~"), "unravel-ai"),  # ~/unravel-ai
            os.path.abspath(".")  # Current directory
        ]
        
        for path in candidates:
            if os.path.exists(os.path.join(path, "setup.sh")) or os.path.exists(os.path.join(path, "ingestion.py")):
                unravel_dir = path
                break
    
    if not unravel_dir:
        print("Unravel AI directory not found. Please specify with --unravel-dir")
        sys.exit(1)
    
    print(f"Patching Unravel AI in: {unravel_dir}")
    
    # Create directories
    os.makedirs(os.path.join(unravel_dir, "src", "core"), exist_ok=True)
    os.makedirs(os.path.join(unravel_dir, "scripts"), exist_ok=True)
    
    # Apply patches
    success = patch_ingestion_py(unravel_dir)
    success &= patch_config_py(unravel_dir)
    success &= patch_setup_sh(unravel_dir)
    
    # Copy LLM module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llm_file = os.path.join(script_dir, "final-llm-integration.py")
    
    if os.path.exists(llm_file):
        dst_path = os.path.join(unravel_dir, "src", "core", "llm_interface.py")
        shutil.copy2(llm_file, dst_path)
        print(f"Copied LLM module to: {dst_path}")
    else:
        print("LLM module not found. Please run: scripts/setup_llm.sh")
    
    # Copy setup script
    setup_llm_sh = os.path.join(script_dir, "install-llm-deps.sh")
    if os.path.exists(setup_llm_sh):
        dst_path = os.path.join(unravel_dir, "scripts", "setup_llm.sh")
        shutil.copy2(setup_llm_sh, dst_path)
        os.chmod(dst_path, 0o755)
        print(f"Copied LLM setup script to: {dst_path}")
    
    if success:
        print("Unravel AI successfully patched for LLM integration")
        print("Run: bash scripts/setup_llm.sh to complete the setup")
    else:
        print("Some patches failed. Please check the logs.")

if __name__ == "__main__":
    main()
