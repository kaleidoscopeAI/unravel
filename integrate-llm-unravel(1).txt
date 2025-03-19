                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                
                if not gguf_files:
                    logger.error(f"No GGUF files found in {model_id}")
                    return None
                
                # Sort by size suffix (Q4_K_M is usually good balance)
                # Prioritize: Q4_K_M > Q4_K > Q5_K_M > Q5_K > Q6_K > Q8_0
                def model_priority(filename):
                    if "Q4_K_M" in filename: return 0
                    if "Q4_K" in filename: return 1
                    if "Q5_K_M" in filename: return 2
                    if "Q5_K" in filename: return 3
                    if "Q6_K" in filename: return 4
                    if "Q8_0" in filename: return 5
                    return 6
                
                gguf_files.sort(key=model_priority)
                
                model_file = gguf_files[0]
                logger.info(f"Downloading {model_id}/{model_file}...")
                model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=model_file,
                    local_dir=model_dir
                )
                return model_path
        
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None
    
    def get_api_client(self, provider: LLMProvider, model_id: str, 
                  api_base: Optional[str] = None) -> LLMAPIClient:
        """
        Get or create an API client
        
        Args:
            provider: LLM provider
            model_id: Model ID or name
            api_base: Base URL for API
            
        Returns:
            API client
        """
        client_key = f"{provider}_{model_id}_{api_base}"
        
        if client_key not in self.api_clients:
            self.api_clients[client_key] = LLMAPIClient(
                provider=provider,
                model_id=model_id,
                api_base=api_base
            )
        
        return self.api_clients[client_key]
    
    def cleanup(self):
        """Clean up resources"""
        # Terminate all processes
        for process_id, process in list(self.active_processes.items()):
            try:
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating process {process_id}")
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating process {process_id}: {str(e)}")
                try:
                    process.kill()
                except:
                    pass
        
        self.active_processes.clear()
        self.api_clients.clear()


class CodeAnalyzer:
    """Analyzes code using LLM capabilities"""
    
    def __init__(self, 
                model_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                model_file: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                provider: Union[str, LLMProvider] = LLMProvider.LLAMACPP_API,
                cache_dir: Optional[str] = None):
        """
        Initialize the code analyzer
        
        Args:
            model_id: Model ID or name
            model_file: Specific model file to use
            provider: LLM provider
            cache_dir: Directory to cache models
        """
        # Normalize provider
        if isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider)
            except ValueError:
                logger.warning(f"Invalid provider '{provider}', falling back to llamacpp_api")
                self.provider = LLMProvider.LLAMACPP_API
        else:
            self.provider = provider
        
        self.model_id = model_id
        self.model_file = model_file
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/unravel-ai/models")
        
        # Initialize manager
        self.manager = LLMModuleManager(cache_dir=self.cache_dir)
        
        # API client
        self.api_client = None
        
        # Setup flag
        self.is_setup = False
    
    def setup(self) -> bool:
        """
        Set up the analyzer
        
        Returns:
            Success status
        """
        if self.is_setup:
            return True
        
        try:
            # Determine flow based on provider
            if self.provider == LLMProvider.LLAMACPP_API:
                # Download model if needed
                model_path = None
                if os.path.exists(self.model_file):
                    model_path = self.model_file
                else:
                    model_path = self.manager.download_model(self.model_id, self.model_file)
                
                if not model_path:
                    logger.error("Failed to download model")
                    return False
                
                # Start server
                if not self.manager.ensure_llamacpp_server(model_path):
                    logger.error("Failed to start llama.cpp server")
                    return False
                
                # Create API client
                self.api_client = self.manager.get_api_client(self.provider, self.model_id)
            
            elif self.provider == LLMProvider.OLLAMA:
                # Check if Ollama is running
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code != 200:
                        logger.error("Ollama API not available")
                        return False
                    
                    # Check if model is available
                    models = response.json().get("models", [])
                    model_names = [model.get("name") for model in models]
                    
                    if self.model_id not in model_names:
                        logger.warning(f"Model {self.model_id} not found in Ollama, trying to pull it")
                        # Try to pull the model
                        pull_response = requests.post(
                            "http://localhost:11434/api/pull",
                            json={"name": self.model_id}
                        )
                        if pull_response.status_code != 200:
                            logger.error(f"Failed to pull model {self.model_id}")
                            return False
                
                except Exception as e:
                    logger.error(f"Error checking Ollama: {str(e)}")
                    return False
                
                # Create API client
                self.api_client = self.manager.get_api_client(self.provider, self.model_id)
            
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return False
            
            self.is_setup = True
            return True
        
        except Exception as e:
            logger.error(f"Error setting up analyzer: {str(e)}")
            return False
    
    def analyze_code(self, 
                    source_code: Union[str, Dict[str, str]],
                    task: Union[str, AnalysisTask] = AnalysisTask.SPECIFICATION,
                    custom_instructions: Optional[str] = None) -> str:
        """
        Analyze source code
        
        Args:
            source_code: Source code string or dict of filename to content
            task: Analysis task or custom task name
            custom_instructions: Custom instructions for the LLM
            
        Returns:
            Analysis result
        """
        # Ensure setup
        if not self.is_setup:
            if not self.setup():
                return "Failed to set up analyzer"
        
        # Normalize source code format
        if isinstance(source_code, str):
            source_code = {"main.txt": source_code}
        
        # Build system prompt based on task
        system_prompt = "You are an expert code analyzer for the Unravel AI system. "
        system_prompt += "Your task is to analyze code with extreme precision and technical depth."
        
        # Build user prompt
        if isinstance(task, str) and task.startswith("custom:"):
            user_prompt = task[7:]  # Remove 'custom:' prefix
        elif custom_instructions:
            user_prompt = custom_instructions
        else:
            # Use predefined task
            task_name = task.value if isinstance(task, AnalysisTask) else task
            
            if task_name == AnalysisTask.SPECIFICATION.value:
                user_prompt = "Create a detailed specification document based on this code. Include: "
                user_prompt += "functionality description, input/output formats, data structures, algorithms used, "
                user_prompt += "dependencies, performance characteristics, and assumptions. Format as Markdown."
            
            elif task_name == AnalysisTask.DOCUMENTATION.value:
                user_prompt = "Generate comprehensive documentation for the following code. Include: "
                user_prompt += "overall purpose, function descriptions, class descriptions, parameter details, "
                user_prompt += "return value explanations, and usage examples. Format the documentation in Markdown."
            
            elif task_name == AnalysisTask.VULNERABILITY.value:
                user_prompt = "Analyze the following code for security vulnerabilities, including but not limited to: "
                user_prompt += "SQL injection, XSS, buffer overflows, memory leaks, race conditions, insecure cryptography, "
                user_prompt += "hardcoded credentials, and any other security issues. Provide a detailed report with "
                user_prompt += "vulnerabilities found, their severity, and recommendations for fixing them."
            
            elif task_name == AnalysisTask.REFACTORING.value:
                user_prompt = "Analyze this code and suggest refactoring improvements. Focus on: code organization, "
                user_prompt += "naming conventions, modularity, performance optimization, error handling, and adherence "
                user_prompt += "to best practices. Provide specific code examples for each suggestion."
            
            elif task_name == AnalysisTask.ALGORITHM_DETECTION.value:
                user_prompt = "Identify and analyze all algorithms in this code. For each algorithm, provide: name, "
                user_prompt += "purpose, implementation details, time complexity, space complexity, edge cases, and "
                user_prompt += "optimization opportunities."
            
            else:
                user_prompt = "Analyze the following code and provide comprehensive insights about its structure, "
                user_prompt += "functionality, potential issues, and opportunities for improvement."
        
        # Add code to prompt
        user_prompt += "\n\nHere's the code to analyze:\n\n"
        
        for filename, code in source_code.items():
            # Determine language from file extension
            language = "text"
            if "." in filename:
                ext = filename.split(".")[-1].lower()
                language_map = {
                    "py": "python", "js": "javascript", "ts": "typescript", 
                    "java": "java", "c": "c", "cpp": "cpp", "cs": "csharp",
                    "go": "go", "rs": "rust", "php": "php", "rb": "ruby",
                    "sh": "bash", "html": "html", "css": "css", "sql": "sql"
                }
                language = language_map.get(ext, "text")
            
            user_prompt += f"File: {filename}\n```{language}\n{code}\n```\n\n"
        
        # Create messages
        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", user_prompt)
        ]
        
        # Configure generation options
        options = LLMRequestOptions(
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            repetition_penalty=1.1
        )
        
        # Generate response
        try:
            response = self.api_client.generate(messages, options)
            return response.content
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return f"Error analyzing code: {str(e)}"
    
    def analyze_files(self, file_paths: List[str], task: Union[str, AnalysisTask] = AnalysisTask.SPECIFICATION) -> Dict[str, str]:
        """
        Analyze multiple files
        
        Args:
            file_paths: List of file paths to analyze
            task: Analysis task or custom task name
            
        Returns:
            Dictionary of filename to analysis results
        """
        # Ensure setup
        if not self.is_setup:
            if not self.setup():
                return {"error": "Failed to set up analyzer"}
        
        results = {}
        
        # Process files
        for file_path in file_paths:
            try:
                # Read file
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Analyze file
                result = self.analyze_code(
                    {os.path.basename(file_path): content},
                    task=task
                )
                
                results[os.path.basename(file_path)] = result
            
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {str(e)}")
                results[os.path.basename(file_path)] = f"Error: {str(e)}"
        
        # Create summary if multiple files
        if len(file_paths) > 1:
            try:
                # Build summary prompt
                summary_prompt = "You've analyzed multiple files from a software project. "
                summary_prompt += "Provide a comprehensive summary of the entire project based on these analyses:\n\n"
                
                for file_name, analysis in results.items():
                    # Limit length to avoid token limits
                    analysis_preview = analysis[:500] + "..." if len(analysis) > 500 else analysis
                    summary_prompt += f"Analysis of {file_name}:\n{analysis_preview}\n\n"
                
                summary_prompt += "\nBased on these individual analyses, provide a unified overview of the entire "
                summary_prompt += "software, including its architecture, key components, data flow, potential "
                summary_prompt += "security issues, and overall quality assessment."
                
                # Generate summary
                messages = [
                    LLMMessage("system", "You are an expert software architect for the Unravel AI system."),
                    LLMMessage("user", summary_prompt)
                ]
                
                options = LLMRequestOptions(
                    temperature=0.3,
                    max_tokens=4096,
                    top_p=0.95
                )
                
                summary_response = self.api_client.generate(messages, options)
                results["_summary"] = summary_response.content
            
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                results["_summary"] = f"Error generating summary: {str(e)}"
        
        return results
    
    def close(self):
        """Clean up resources"""
        self.manager.cleanup()


# Main class for integration with Unravel AI
class UnravelAIAnalyzer:
    """Main analyzer for Unravel AI using LLM capabilities"""
    
    def __init__(self, 
                model_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                model_file: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                provider: Union[str, LLMProvider] = LLMProvider.LLAMACPP_API,
                cache_dir: Optional[str] = None):
        """
        Initialize the analyzer
        
        Args:
            model_id: Model ID or name
            model_file: Specific model file to use
            provider: LLM provider
            cache_dir: Directory to cache models
        """
        self.analyzer = CodeAnalyzer(
            model_id=model_id,
            model_file=model_file,
            provider=provider,
            cache_dir=cache_dir
        )
    
    def setup(self, force_download: bool = False) -> bool:
        """
        Set up the analyzer
        
        Args:
            force_download: Whether to force re-download the model
            
        Returns:
            Success status
        """
        return self.analyzer.setup()
    
    def analyze_file(self, file_path: str, analysis_type: str = "specification") -> str:
        """
        Analyze a single file
        
        Args:
            file_path: Path to the file
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        try:
            # Read the file
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Analyze content
            return self.analyzer.analyze_code(
                {os.path.basename(file_path): content},
                task=analysis_type
            )
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return f"Error analyzing file: {str(e)}"
    
    def analyze_software(self, decompiled_files: List[str], analysis_type: str = "specification") -> Dict[str, str]:
        """
        Analyze a set of decompiled files
        
        Args:
            decompiled_files: List of paths to decompiled files
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary of file to analysis results
        """
        return self.analyzer.analyze_files(decompiled_files, task=analysis_type)
    
    def cleanup(self):
        """Clean up resources"""
        self.analyzer.close()

# Test function
if __name__ == "__main__":
    # CLI implementation
    import argparse
    
    parser = argparse.ArgumentParser(description="Unravel AI LLM Module")
    parser.add_argument("--model", default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", help="Model ID")
    parser.add_argument("--model-file", default="mistral-7b-instruct-v0.2.Q4_K_M.gguf", help="Model file")
    parser.add_argument("--provider", default="llamacpp_api", choices=["llamacpp_api", "ollama"], help="LLM provider")
    parser.add_argument("--cache-dir", help="Cache directory for models")
    parser.add_argument("--analyze", help="File or directory to analyze")
    parser.add_argument("--type", default="specification", help="Analysis type")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Create analyzer
        analyzer = UnravelAIAnalyzer(
            model_id=args.model,
            model_file=args.model_file,
            provider=args.provider,
            cache_dir=args.cache_dir
        )
        
        try:
            # Setup analyzer
            if not analyzer.setup():
                print("Failed to set up analyzer")
                sys.exit(1)
            
            # Determine if file or directory
            path = Path(args.analyze)
            if path.is_file():
                print(f"Analyzing file: {path}")
                result = analyzer.analyze_file(str(path), args.type)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(result)
                    print(f"Results saved to {args.output}")
                else:
                    print(result)
            
            elif path.is_dir():
                print(f"Analyzing directory: {path}")
                files = [str(f) for f in path.glob("**/*") if f.is_file()]
                results = analyzer.analyze_software(files, args.type)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        for file_name, result in results.items():
                            f.write(f"# Analysis of {file_name}\n\n")
                            f.write(result)
                            f.write("\n\n---\n\n")
                    print(f"Results saved to {args.output}")
                else:
                    # Print summary if available
                    if "_summary" in results:
                        print(results["_summary"])
                    else:
                        # Print number of files analyzed
                        print(f"Analyzed {len(results)} files")
            else:
                print(f"Path not found: {args.analyze}")
        
        finally:
            # Clean up
            analyzer.cleanup()
    else:
        print("No analysis target specified. Use --analyze to specify a file or directory.")
        sys.exit(1)
EOF

# Create the installation script for llama.cpp
cat > "${REPO_DIR}/scripts/install_llamacpp.sh" << 'EOF'
#!/bin/bash
# Install llama.cpp for Unravel AI
set -e

# Configuration
BUILD_DIR="${HOME}/.cache/unravel-ai/build"
INSTALL_DIR="${HOME}/.local/bin"

# Determine number of threads for compilation
if command -v nproc &> /dev/null; then
    NUM_THREADS=$(nproc)
elif command -v sysctl &> /dev/null && sysctl -n hw.ncpu &> /dev/null; then
    NUM_THREADS=$(sysctl -n hw.ncpu)
else
    NUM_THREADS=4
fi

# Install system dependencies
if [ -f /etc/debian_version ]; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git
elif [ -f /etc/redhat-release ]; then
    sudo yum install -y gcc-c++ cmake git
elif [ -f /etc/arch-release ]; then
    sudo pacman -S --needed base-devel cmake git
elif [ "$(uname)" == "Darwin" ]; then
    if ! command -v brew &> /dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install cmake git
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Clone and build llama.cpp
cd "${BUILD_DIR}"
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
mkdir -p build
cd build

CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF"

# Check for AVX2 support for performance optimization
if grep -q "avx2" /proc/cpuinfo 2>/dev/null || sysctl -a | grep -q "hw.optional.avx2_0" 2>/dev/null; then
    echo "Enabling AVX2 optimizations"
else
    echo "Disabling CPU optimizations for better compatibility"
    CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_FMA=OFF"
fi

# Configure and build
cmake .. ${CMAKE_ARGS}
cmake --build . --config Release -j ${NUM_THREADS}

# Install binaries
mkdir -p "${INSTALL_DIR}"
cp -f bin/llama-server "${INSTALL_DIR}/"
cp -f bin/main "${INSTALL_DIR}/llama-cpp"
ln -sf "${INSTALL_DIR}/llama-server" "${INSTALL_DIR}/server"

# Add to PATH if needed
if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
    export PATH="${INSTALL_DIR}:$PATH"
fi

echo "llama.cpp installed successfully"
EOF
chmod +x "${REPO_DIR}/scripts/install_llamacpp.sh"

# Create a patch for ingestion.py to add LLM support
cat > "${REPO_DIR}/scripts/patch_ingestion.py" << 'EOF'
#!/usr/bin/env python3
"""
Patch ingestion.py to add LLM support
"""
import os
import re
import sys
import shutil

def patch_file(file_path, patterns):
    """
    Apply multiple regex patterns to a file
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Apply each pattern
    modified = False
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        if new_content != content:
            content = new_content
            modified = True
    
    # Write modified content
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully patched: {file_path}")
        return True
    else:
        print(f"No changes needed for: {file_path}")
        return False

def patch_ingestion():
    """
    Patch the ingestion.py file to add LLM support
    """
    # Find ingestion.py
    candidates = [
        "ingestion.py",
        "src/core/ingestion.py",
        "../ingestion.py",
        "../src/core/ingestion.py"
    ]
    
    ingestion_path = None
    for path in candidates:
        if os.path.exists(path):
            ingestion_path = os.path.abspath(path)
            break
    
    if not ingestion_path:
        print("Error: Could not find ingestion.py")
        return False
    
    # Define patterns to apply
    patterns = [
        # 1. Add import for LLM analyzer
        (
            r"(from dataclasses import dataclass.*?\n)",
            r"\1import importlib.util\n"
        ),
        # 2. Add LLM check in SpecGenerator.__init__
        (
            r"(class SpecGenerator.*?def __init__.*?self\.specs_dir = os\.path\.join\(self\.work_dir, \"specs\"\)\n)",
            r"\1        # Check if LLM module is available\n        self.use_llm = False\n        if importlib.util.find_spec(\"src.core.llm_interface\") is not None:\n            self.use_llm = True\n            logger.info(\"LLM module available for specification generation\")\n"
        ),
        # 3. Modify generate_specifications to check for LLM
        (
            r"(def generate_specifications\(self, decompiled_files: List\[str\]\) -> List\[str\]:.*?)(\s+if not decompiled_files:)",
            r"\1        # Use LLM if available\n        if self.use_llm:\n            try:\n                from src.core.llm_interface import UnravelAIAnalyzer\n                analyzer = UnravelAIAnalyzer()\n                \n                if analyzer.setup():\n                    logger.info(\"Using LLM for enhanced specification generation\")\n                    results = analyzer.analyze_software(decompiled_files, analysis_type=\"specification\")\n                    \n                    # Create spec files from results\n                    for file_name, result in results.items():\n                        if file_name == \"_summary\":\n                            # Create combined spec\n                            summary_path = os.path.join(self.specs_dir, \"combined_spec.md\")\n                            with open(summary_path, 'w') as f:\n                                f.write(\"# Software Specification\\n\\n\")\n                                f.write(\"This document contains specifications extracted from the decompiled software.\\n\\n\")\n                                f.write(result)\n                            spec_files.append(summary_path)\n                        else:\n                            # Create individual spec\n                            file_path = next((f for f in decompiled_files if os.path.basename(f) == file_name), None)\n                            if file_path:\n                                spec_path = os.path.join(self.specs_dir, f\"{os.path.basename(file_path)}.spec.md\")\n                                with open(spec_path, 'w') as f:\n                                    f.write(f\"# {os.path.basename(file_path)} Specification\\n\\n\")\n                                    f.write(result)\n                                spec_files.append(spec_path)\n                    \n                    # Create specialized specifications\n                    try:\n                        api_results = analyzer.analyze_software(decompiled_files, analysis_type=\"documentation\")\n                        api_path = os.path.join(self.specs_dir, \"api_documentation.md\")\n                        with open(api_path, 'w') as f:\n                            f.write(\"# API Documentation\\n\\n\")\n                            f.write(\"This document describes the public API of the software.\\n\\n\")\n                            if \"_summary\" in api_results:\n                                f.write(api_results[\"_summary\"])\n                        spec_files.append(api_path)\n                        \n                        # Add more specialized specs as needed\n                    except Exception as e:\n                        logger.error(f\"Error generating specialized specs: {str(e)}\")\n                    \n                    # Clean up\n                    analyzer.cleanup()\n                    return spec_files\n                else:\n                    logger.warning(\"LLM setup failed, falling back to standard specification generation\")\n            except Exception as e:\n                logger.error(f\"Error using LLM for specification generation: {str(e)}\")\n                logger.warning(\"Falling back to standard specification generation\")\n\2"
        )
    ]
    
    return patch_file(ingestion_path, patterns)

if __name__ == "__main__":
    patch_ingestion()
EOF
chmod +x "${REPO_DIR}/scripts/patch_ingestion.py"

# Create a configuration file
mkdir -p "${HOME}/.config/unravel-ai"
cat > "${HOME}/.config/unravel-ai/llm_config.json" << EOF
{
    "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "model_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "provider": "llamacpp_api",
    "cache_dir": "${HOME}/.cache/unravel-ai/models",
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "system_prompt": "You are an expert code analyzer for the Unravel AI system. Your task is to analyze code with extreme precision and technical depth."
}
EOF

# Create a test script
cat > "${REPO_DIR}/test_llm.#!/bin/bash
# Full automatic integration of LLM capabilities into Unravel AI
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PYTHON_CMD="$(command -v python3 || command -v python)"
FORCE_INSTALL=0
SKIP_DEPS=0

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_INSTALL=1; shift ;;
        --skip-deps) SKIP_DEPS=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create needed directories
mkdir -p "${REPO_DIR}/src/core"
mkdir -p "${REPO_DIR}/scripts"
mkdir -p "${HOME}/.cache/unravel-ai/models"
mkdir -p "${HOME}/.config/unravel-ai"

# Install Python dependencies if needed
if [ $SKIP_DEPS -eq 0 ]; then
    echo "Installing required Python packages..."
    ${PYTHON_CMD} -m pip install --upgrade pip
    ${PYTHON_CMD} -m pip install huggingface_hub requests numpy psutil setuptools wheel
    ${PYTHON_CMD} -m pip install llama-cpp-python
fi

# Create the LLM interface module
cat > "${REPO_DIR}/src/core/llm_interface.py" << 'EOF'
#!/usr/bin/env python3
"""
Unravel AI - Comprehensive LLM Integration with llama.cpp for CPU inference
"""
import os
import sys
import re
import json
import time
import asyncio
import logging
import hashlib
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
import subprocess
import psutil
import concurrent.futures
import requests
import numpy as np
from enum import Enum
import importlib.util
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Auto-detect availability of optional dependencies
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_HUGGINGFACE_HUB = importlib.util.find_spec("huggingface_hub") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_CTRANSFORMERS = importlib.util.find_spec("ctransformers") is not None
HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    LLAMACPP_API = "llamacpp_api"
    LLAMACPP_PYTHON = "llamacpp_python"
    CTRANSFORMERS = "ctransformers"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL_API = "local_api"

class AnalysisTask(str, Enum):
    """Types of code analysis tasks"""
    SPECIFICATION = "specification"
    DOCUMENTATION = "documentation"
    VULNERABILITY = "vulnerability"
    REFACTORING = "refactoring"
    FUNCTION_SUMMARY = "function_summary"
    CLASS_SUMMARY = "class_summary"
    ALGORITHM_DETECTION = "algorithm_detection"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    CUSTOM = "custom"

class LLMMessage:
    """Represents a message in a conversation with an LLM"""
    
    def __init__(self, role: str, content: str):
        """
        Initialize a message
        
        Args:
            role: Message role ("system", "user", "assistant")
            content: Message content
        """
        self.role = role
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API requests"""
        return {"role": self.role, "content": self.content}

class LLMRequestOptions:
    """Options for an LLM request"""
    
    def __init__(self, 
                 temperature: float = 0.2,
                 top_p: float = 0.95,
                 top_k: int = 40,
                 max_tokens: int = 4096,
                 stream: bool = False,
                 stop_sequences: Optional[List[str]] = None,
                 repetition_penalty: float = 1.1,
                 num_threads: Optional[int] = None):
        """
        Initialize request options
        
        Args:
            temperature: Temperature for sampling
            top_p: Top-p probability threshold
            top_k: Top-k sampling threshold
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            stop_sequences: Sequences to stop generation
            repetition_penalty: Penalty for token repetition
            num_threads: Number of CPU threads to use
        """
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stream = stream
        self.stop_sequences = stop_sequences or []
        self.repetition_penalty = repetition_penalty
        self.num_threads = num_threads or max(1, psutil.cpu_count(logical=False) - 1)

class LLMUsage:
    """Tracks token usage"""
    
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """
        Initialize token usage
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

class LLMResponse:
    """Response from an LLM"""
    
    def __init__(self, content: str, usage: LLMUsage, model_id: str, finish_reason: str = "stop"):
        """
        Initialize a response
        
        Args:
            content: Response content
            usage: Token usage statistics
            model_id: ID of the model used
            finish_reason: Reason for finishing generation
        """
        self.content = content
        self.usage = usage
        self.model_id = model_id
        self.finish_reason = finish_reason

class TokenCounter:
    """Approximate token counter for LLMs"""
    
    def __init__(self):
        """Initialize the token counter"""
        self.words_per_token = 0.75  # Approximate ratio
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string (approximate)
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        # Simple approximation based on word count
        words = len(text.split())
        return int(words / self.words_per_token)

class PromptBuilder:
    """Builds effective prompts for different LLM models"""
    
    def __init__(self, model_id: str):
        """
        Initialize the prompt builder
        
        Args:
            model_id: Model ID or name
        """
        self.model_id = model_id.lower()
        self.token_counter = TokenCounter()
    
    def build_messages(self, system_prompt: str, user_prompt: str) -> List[LLMMessage]:
        """
        Build messages for a conversation
        
        Args:
            system_prompt: System instruction
            user_prompt: User query or input
            
        Returns:
            List of messages
        """
        return [
            LLMMessage("system", system_prompt),
            LLMMessage("user", user_prompt)
        ]
    
    def format_prompt(self, messages: List[LLMMessage]) -> str:
        """
        Format messages into a prompt string based on the model
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted prompt
        """
        # Llama 2 Chat format
        if "llama-2" in self.model_id and "chat" in self.model_id:
            return self._format_llama2_chat(messages)
        # Mistral format
        elif "mistral" in self.model_id:
            return self._format_mistral(messages)
        # CodeLlama format
        elif "codellama" in self.model_id:
            return self._format_codellama(messages)
        # Phi format
        elif "phi" in self.model_id:
            return self._format_phi(messages)
        # Default to simple format
        else:
            return self._format_simple(messages)
    
    def _format_llama2_chat(self, messages: List[LLMMessage]) -> str:
        """Format messages for Llama 2 Chat models"""
        system_prompt = "You are a helpful, respectful and honest assistant."
        formatted_messages = []
        
        # Extract system prompt if present
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                break
        
        # Build conversation with Llama 2 format
        conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        for i, msg in enumerate(messages):
            if msg.role == "system":
                continue
            elif msg.role == "user":
                if i > 0 and conversation and not conversation.endswith("[INST] "):
                    conversation += " [/INST] "
                if i > 0 and not conversation.endswith("[INST] "):
                    conversation += "[INST] "
                conversation += msg.content
            elif msg.role == "assistant":
                conversation += f" [/INST] {msg.content} </s>"
                if i < len(messages) - 1:
                    conversation += "<s>"
        
        # Close the final user message if needed
        if conversation and not conversation.endswith("[/INST] ") and not conversation.endswith("</s>"):
            conversation += " [/INST] "
        
        return conversation
    
    def _format_mistral(self, messages: List[LLMMessage]) -> str:
        """Format messages for Mistral models"""
        formatted = ""
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        
        # Start with the system message if present
        if system_content:
            formatted += f"<s>[INST] {system_content} [/INST]</s>"
        
        # Add messages
        for i, msg in enumerate(messages):
            if msg.role == "system":
                continue
            elif msg.role == "user":
                formatted += f"<s>[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content}</s>"
        
        return formatted
    
    def _format_codellama(self, messages: List[LLMMessage]) -> str:
        """Format messages for CodeLlama models"""
        # CodeLlama uses the same format as Llama 2 Chat
        return self._format_llama2_chat(messages)
    
    def _format_phi(self, messages: List[LLMMessage]) -> str:
        """Format messages for Phi models"""
        formatted = ""
        
        # Process messages
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"Human: {msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n\n"
        
        # Add assistant prefix for generation
        if messages[-1].role != "assistant":
            formatted += "Assistant: "
        
        return formatted
    
    def _format_simple(self, messages: List[LLMMessage]) -> str:
        """Simple format for any model"""
        formatted = ""
        
        # Extract system message
        system_msg = next((msg for msg in messages if msg.role == "system"), None)
        if system_msg:
            formatted += f"### System:\n{system_msg.content}\n\n"
        
        # Add user and assistant messages
        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                formatted += f"### User:\n{msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"### Assistant:\n{msg.content}\n\n"
        
        # Add assistant prompt for generation
        if messages[-1].role != "assistant":
            formatted += "### Assistant:\n"
        
        return formatted

class LLMAPIClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(self, 
                 provider: LLMProvider,
                 model_id: str,
                 api_base: Optional[str] = None,
                 timeout: int = 120):
        """
        Initialize the API client
        
        Args:
            provider: LLM provider
            model_id: Model ID or name
            api_base: Base URL for API
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model_id = model_id
        self.timeout = timeout
        
        # Set API base URL
        if api_base:
            self.api_base = api_base
        elif provider == LLMProvider.LLAMACPP_API:
            self.api_base = "http://localhost:8080"
        elif provider == LLMProvider.OLLAMA:
            self.api_base = "http://localhost:11434/api"
        elif provider == LLMProvider.LOCAL_API:
            self.api_base = "http://localhost:8000"
        else:
            self.api_base = None
        
        self.prompt_builder = PromptBuilder(model_id)
        self.token_counter = TokenCounter()
    
    def generate(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """
        Generate a response using the API
        
        Args:
            messages: List of messages
            options: Generation options
            
        Returns:
            LLM response
        """
        # Handle each provider type
        if self.provider == LLMProvider.LLAMACPP_API:
            return self._generate_llamacpp_api(messages, options)
        elif self.provider == LLMProvider.OLLAMA:
            return self._generate_ollama(messages, options)
        elif self.provider == LLMProvider.LOCAL_API:
            return self._generate_local_api(messages, options)
        else:
            raise ValueError(f"Unsupported provider for API client: {self.provider}")
    
    def _generate_llamacpp_api(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """Generate using llama.cpp HTTP server"""
        # Format prompt
        prompt = self.prompt_builder.format_prompt(messages)
        
        # Count tokens
        prompt_tokens = self.token_counter.count_tokens(prompt)
        
        # Prepare request
        request_data = {
            "prompt": prompt,
            "n_predict": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "repeat_penalty": options.repetition_penalty,
            "stop": options.stop_sequences if options.stop_sequences else None,
            "stream": options.stream,
            "n_threads": options.num_threads
        }
        
        # Make request
        try:
            if options.stream:
                full_response = ""
                response = requests.post(
                    f"{self.api_base}/completion",
                    json=request_data,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    chunk = json.loads(line)
                    content = chunk.get("content", "")
                    full_response += content
                    
                    # Check if generation is done
                    if chunk.get("stop", False):
                        break
            else:
                response = requests.post(
                    f"{self.api_base}/completion",
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                full_response = data.get("content", "")
        
        except Exception as e:
            logger.error(f"Error calling llama.cpp API: {str(e)}")
            raise RuntimeError(f"LLM API error: {str(e)}")
        
        # Strip any prefix matching the prompt
        if full_response.startswith(prompt):
            full_response = full_response[len(prompt):]
        
        # Count completion tokens
        completion_tokens = self.token_counter.count_tokens(full_response)
        
        # Create usage metrics
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        return LLMResponse(
            content=full_response.strip(),
            usage=usage,
            model_id=self.model_id
        )
    
    def _generate_ollama(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """Generate using Ollama API"""
        # Prepare request data
        formatted_messages = [msg.to_dict() for msg in messages]
        
        request_data = {
            "model": self.model_id,
            "messages": formatted_messages,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "top_k": options.top_k,
                "num_predict": options.max_tokens,
                "repeat_penalty": options.repetition_penalty
            },
            "stream": options.stream
        }
        
        # Count prompt tokens
        prompt_text = "\n".join([msg.content for msg in messages])
        prompt_tokens = self.token_counter.count_tokens(prompt_text)
        
        # Make request
        try:
            if options.stream:
                full_response = ""
                response = requests.post(
                    f"{self.api_base}/chat",
                    json=request_data,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(f"Ollama API error: {chunk['error']}")
                    
                    if "message" in chunk:
                        message_content = chunk["message"]["content"]
                        full_response += message_content
                    
                    # Check if generation is done
                    if chunk.get("done", False):
                        break
            else:
                response = requests.post(
                    f"{self.api_base}/chat",
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if "error" in data:
                    raise RuntimeError(f"Ollama API error: {data['error']}")
                
                full_response = data["message"]["content"]
        
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise RuntimeError(f"LLM API error: {str(e)}")
        
        # Count completion tokens
        completion_tokens = self.token_counter.count_tokens(full_response)
        
        # Create usage metrics
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        return LLMResponse(
            content=full_response.strip(),
            usage=usage,
            model_id=self.model_id
        )
    
    def _generate_local_api(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """Generate using local API (generic implementation)"""
        # Format prompt
        prompt = self.prompt_builder.format_prompt(messages)
        
        # Count tokens
        prompt_tokens = self.token_counter.count_tokens(prompt)
        
        # Prepare request
        request_data = {
            "prompt": prompt,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "stop": options.stop_sequences if options.stop_sequences else None,
            "stream": options.stream
        }
        
        # Make request
        try:
            if options.stream:
                full_response = ""
                response = requests.post(
                    f"{self.api_base}/generate",
                    json=request_data,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        text = chunk.get("text", "")
                        full_response += text
                    except json.JSONDecodeError:
                        # Handle non-JSON responses
                        full_response += line.decode('utf-8')
            else:
                response = requests.post(
                    f"{self.api_base}/generate",
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                full_response = data.get("text", "")
        
        except Exception as e:
            logger.error(f"Error calling local API: {str(e)}")
            raise RuntimeError(f"LLM API error: {str(e)}")
        
        # Count completion tokens
        completion_tokens = self.token_counter.count_tokens(full_response)
        
        # Create usage metrics
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        return LLMResponse(
            content=full_response.strip(),
            usage=usage,
            model_id=self.model_id
        )

class LLMModuleManager:
    """Manager for LLM modules and models"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the manager
        
        Args:
            cache_dir: Directory to cache models
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/unravel-ai/models")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.active_processes = {}
        self.api_clients = {}
    
    def ensure_llamacpp_server(self, model_path: str, port: int = 8080, n_ctx: int = 2048, 
                               n_threads: Optional[int] = None) -> bool:
        """
        Ensure llama.cpp server is running
        
        Args:
            model_path: Path to the model file
            port: Server port
            n_ctx: Context size
            n_threads: Number of threads to use
            
        Returns:
            Whether the server is running
        """
        server_id = f"llamacpp_{port}"
        
        # Check if already running
        if server_id in self.active_processes:
            process = self.active_processes[server_id]
            if process.poll() is None:  # Process is still running
                return True
            else:
                # Process has terminated
                del self.active_processes[server_id]
        
        # Check if another server is running on this port
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"llama.cpp server already running on port {port}")
                return True
        except:
            # Server not running
            pass
        
        # Find llama-server executable
        server_executable = self._find_llamacpp_executable()
        if not server_executable:
            logger.error("llama-server executable not found, attempting to install...")
            if not self._install_llamacpp():
                logger.error("Failed to install llama.cpp")
                return False
            server_executable = self._find_llamacpp_executable()
            if not server_executable:
                logger.error("Still cannot find llama-server executable")
                return False
        
        # Set number of threads
        if n_threads is None:
            n_threads = max(1, psutil.cpu_count(logical=False) - 1)
        
        # Start server
        logger.info(f"Starting llama.cpp server with model {model_path} on port {port}...")
        cmd = [
            server_executable,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(n_ctx),
            "--threads", str(n_threads),
            "--ctx-size", str(n_ctx),
            "--batch-size", "512"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.active_processes[server_id] = process
        
        # Wait for server to start
        max_wait = 30  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"llama.cpp server started successfully on port {port}")
                    return True
            except:
                # Server not ready yet
                time.sleep(1)
                
                # Check if process has terminated
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Server process terminated: {stderr}")
                    del self.active_processes[server_id]
                    return False
        
        logger.error(f"Timed out waiting for llama.cpp server to start on port {port}")
        return False
    
    def _find_llamacpp_executable(self) -> Optional[str]:
        """Find llama-server executable"""
        # Try common names
        for name in ["llama-server", "server"]:
            # Check in PATH
            path = shutil.which(name)
            if path:
                return path
            
            # Check common locations
            for location in [
                os.path.expanduser("~/.local/bin"),
                "/usr/local/bin",
                "/usr/bin",
                os.path.join(os.path.dirname(sys.executable), "Scripts")  # Windows
            ]:
                candidate = os.path.join(location, name)
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        
        return None
    
    def _install_llamacpp(self) -> bool:
        """Install llama.cpp"""
        try:
            # Create build directory
            build_dir = os.path.join(self.cache_dir, "build_llamacpp")
            os.makedirs(build_dir, exist_ok=True)
            
            # Clone repository
            logger.info("Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"],
                cwd=build_dir,
                check=True
            )
            
            # Build
            logger.info("Building llama.cpp...")
            build_path = os.path.join(build_dir, "llama.cpp", "build")
            os.makedirs(build_path, exist_ok=True)
            
            subprocess.run(
                ["cmake", ".."],
                cwd=build_path,
                check=True
            )
            
            n_threads = max(1, os.cpu_count() or 4)
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "-j", str(n_threads)],
                cwd=build_path,
                check=True
            )
            
            # Install
            logger.info("Installing llama.cpp...")
            install_dir = os.path.expanduser("~/.local/bin")
            os.makedirs(install_dir, exist_ok=True)
            
            # Copy binary
            for src, dst in [
                (os.path.join(build_path, "bin", "llama-server"), os.path.join(install_dir, "llama-server")),
                (os.path.join(build_path, "bin", "main"), os.path.join(install_dir, "llama-cpp"))
            ]:
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    os.chmod(dst, 0o755)  # Make executable
            
            # Create symlink
            server_path = os.path.join(install_dir, "server")
            if not os.path.exists(server_path):
                os.symlink(os.path.join(install_dir, "llama-server"), server_path)
            
            return True
        except Exception as e:
            logger.error(f"Error installing llama.cpp: {e}")
            return False
    
    def download_model(self, model_id: str, model_file: Optional[str] = None) -> Optional[str]:
        """
        Download a model from Hugging Face
        
        Args:
            model_id: Hugging Face model ID
            model_file: Specific model file to download
            
        Returns:
            Path to the downloaded model or None if failed
        """
        if not HAS_HUGGINGFACE_HUB:
            logger.error("huggingface_hub not installed, cannot download model")
            return None
        
        try:
            from huggingface_hub import hf_hub_download
            
            # Create model directory
            model_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)
            
            # Download model file
            if model_file:
                logger.info(f"Downloading {model_id}/{model_file}...")
                model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=model_file,
                    local_dir=model_dir
                )
                return model_path
            else:
                # Try to find a GGUF file
                logger.info(f"Searching for GGUF model in {model_id}...")
                from huggingface_hub import list_repo_files
                
                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.en