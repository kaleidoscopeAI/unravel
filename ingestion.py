#!/usr/bin/env python3
"""
Unravel AI Core Engine
Part 1: Software Ingestion, Decompilation, and Analysis
"""

import os
import sys
import shutil
import tempfile
import subprocess
import logging
import json
import re
import hashlib
import asyncio
import aiohttp
import base64
import zlib
import networkx as nx
import tiktoken
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# File type definitions
class FileType(Enum):
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
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

# Decompilation strategies
class DecompStrategy(Enum):
    RADARE2 = "radare2"
    GHIDRA = "ghidra"
    RETDEC = "retdec"
    IDA = "ida"
    BINARY_NINJA = "binary_ninja"
    CUSTOM = "custom"

@dataclass
class AnalysisResult:
    """Results from analyzing a software artifact"""
    software_id: str
    file_path: str
    file_type: FileType
    status: str
    decompiled_files: List[str] = field(default_factory=list)
    spec_files: List[str] = field(default_factory=list)
    reconstructed_files: List[str] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    graph: Optional[nx.DiGraph] = field(default=None)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert non-serializable types
        result["file_type"] = self.file_type.value
        if self.graph:
            # Convert graph to adjacency list
            result["graph"] = nx.node_link_data(self.graph)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary"""
        # Convert string enum values to actual enums
        if "file_type" in data:
            data["file_type"] = FileType(data["file_type"])
        # Convert adjacency list to graph
        if "graph" in data and data["graph"]:
            graph_data = data.pop("graph")
            graph = nx.node_link_graph(graph_data)
            return cls(**data, graph=graph)
        return cls(**data)

class TokenCounter:
    """Utility for counting tokens in texts for LLM processing"""
    
    def __init__(self, model: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def split_into_chunks(self, text: str, max_tokens: int = 8000, 
                           overlap: int = 500) -> List[str]:
        """Split text into overlapping chunks that fit token limits"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for line in lines:
            line_token_count = self.count_tokens(line)
            
            if current_token_count + line_token_count > max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Calculate overlap
                overlap_start = max(0, len(current_chunk) - self._calculate_overlap_lines(current_chunk, overlap))
                current_chunk = current_chunk[overlap_start:]
                current_token_count = self.count_tokens('\n'.join(current_chunk))
            
            current_chunk.append(line)
            current_token_count += line_token_count
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _calculate_overlap_lines(self, lines: List[str], target_tokens: int) -> int:
        """Calculate how many lines to include in the overlap"""
        total_lines = len(lines)
        token_count = 0
        lines_needed = 0
        
        for i in range(total_lines - 1, -1, -1):
            line_tokens = self.count_tokens(lines[i])
            if token_count + line_tokens > target_tokens:
                break
            
            token_count += line_tokens
            lines_needed += 1
            
            if lines_needed >= total_lines // 2:
                break
                
        return lines_needed

class FileAnalyzer:
    """Analyzes files to extract structure and dependencies"""
    
    def __init__(self):
        self.token_counter = TokenCounter()
    
    def detect_file_type(self, file_path: str) -> FileType:
        """Detect the file type based on extension and content"""
        # Extension mapping
        ext_map = {
            ".exe": FileType.BINARY, ".dll": FileType.BINARY, ".so": FileType.BINARY,
            ".dylib": FileType.BINARY, ".o": FileType.BINARY, ".obj": FileType.BINARY,
            ".js": FileType.JAVASCRIPT, ".mjs": FileType.JAVASCRIPT,
            ".ts": FileType.TYPESCRIPT, ".tsx": FileType.TYPESCRIPT,
            ".py": FileType.PYTHON, ".pyc": FileType.PYTHON,
            ".cpp": FileType.CPP, ".cc": FileType.CPP, ".cxx": FileType.CPP,
            ".c": FileType.C, ".h": FileType.C, ".hpp": FileType.CPP,
            ".cs": FileType.CSHARP, ".java": FileType.JAVA, ".go": FileType.GO,
            ".rs": FileType.RUST, ".asm": FileType.ASSEMBLY, ".s": FileType.ASSEMBLY
        }
        
        # Try by extension first
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ext_map:
            return ext_map[file_ext]
        
        # Try file command
        try:
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            if any(x in output.lower() for x in ["elf", "executable", "binary", "mach-o", "pe32"]):
                return FileType.BINARY
            elif "javascript" in output.lower():
                return FileType.JAVASCRIPT
            elif "python" in output.lower():
                return FileType.PYTHON
            # Add more file type checks here...
        except:
            pass
        
        # Try content analysis
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(4096)  # Read first 4K
                
                if re.search(r'import\s+{.*?}\s+from|require\(|export|=>|function\s+\w+\s*\(', content):
                    return FileType.JAVASCRIPT
                elif re.search(r'import\s+\w+|def\s+\w+\s*\(.*\):|class\s+\w+:', content):
                    return FileType.PYTHON
                elif re.search(r'#include\s+<\w+\.h>|template\s+<typename|std::', content):
                    return FileType.CPP
                # Add more content checks here...
        except:
            pass
        
        # Default to binary if we can't read it as text
        try:
            with open(file_path, 'r') as f:
                f.read(10)  # Try to read as text
            return FileType.UNKNOWN
        except:
            return FileType.BINARY
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a file and extract metadata, structures, and dependencies"""
        file_type = self.detect_file_type(file_path)
        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": file_type.value,
            "file_size": os.path.getsize(file_path),
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": []
        }
        
        # Type-specific analysis
        if file_type == FileType.PYTHON:
            self._analyze_python(file_path, result)
        elif file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT]:
            self._analyze_javascript(file_path, result)
        elif file_type in [FileType.C, FileType.CPP]:
            self._analyze_c_cpp(file_path, result)
        elif file_type == FileType.JAVA:
            self._analyze_java(file_path, result)
        elif file_type == FileType.BINARY:
            self._analyze_binary(file_path, result)
        
        return result
    
    def _analyze_python(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract Python code structure"""
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
            
        # Extract imports
        for match in re.finditer(r'^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+)(?:\s+as\s+(\w+))?)', content, re.MULTILINE):
            if match.group(1):  # from X import Y
                module = match.group(1)
                imports = [name.strip() for name in match.group(2).split(',')]
                for imported in imports:
                    result["imports"].append({"module": module, "name": imported})
            else:  # import X
                module = match.group(3)
                alias = match.group(4)
                result["imports"].append({"module": module, "alias": alias})
        
        # Extract functions
        for match in re.finditer(r'^\s*def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?:', content, re.MULTILINE):
            name = match.group(1)
            params = match.group(2).strip()
            return_type = match.group(3)
            result["functions"].append({
                "name": name,
                "params": params,
                "return_type": return_type
            })
        
        # Extract classes
        for match in re.finditer(r'^\s*class\s+(\w+)(?:\((.*?)\))?:', content, re.MULTILINE):
            name = match.group(1)
            inherits = match.group(2).split(',') if match.group(2) else []
            result["classes"].append({
                "name": name,
                "inherits": [base.strip() for base in inherits if base.strip()]
            })
    
    def _analyze_javascript(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract JavaScript/TypeScript code structure"""
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Extract imports
        for match in re.finditer(r'(?:import\s+{(.*?)}\s+from\s+[\'"](.+?)[\'"]|import\s+(\w+)\s+from\s+[\'"](.+?)[\'"]|require\s*\(\s*[\'"](.+?)[\'"]\s*\))', content):
            if match.group(1) and match.group(2):  # import {X} from "Y"
                names = [n.strip() for n in match.group(1).split(',')]
                module = match.group(2)
                for name in names:
                    result["imports"].append({"name": name, "module": module})
            elif match.group(3) and match.group(4):  # import X from "Y"
                name = match.group(3)
                module = match.group(4)
                result["imports"].append({"name": name, "module": module})
            elif match.group(5):  # require("X")
                module = match.group(5)
                result["imports"].append({"module": module})
        
        # Extract functions
        for match in re.finditer(r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function\s*\(|async\s*(?:function)?\s*\(|\([^)]*\)\s*=>))', content):
            name = match.group(1) or match.group(2)
            result["functions"].append({"name": name})
        
        # Extract classes
        for match in re.finditer(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', content):
            name = match.group(1)
            extends = match.group(2)
            result["classes"].append({
                "name": name,
                "extends": extends
            })
        
        # Extract exports
        for match in re.finditer(r'(?:export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)|module\.exports(?:\.(\w+)|\[[\'"](\w+)[\'"]\]))', content):
            name = match.group(1) or match.group(2) or match.group(3)
            if name:
                result["exports"].append({"name": name})
    
    def _analyze_c_cpp(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract C/C++ code structure"""
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Extract includes
        includes = []
        for match in re.finditer(r'#include\s+[<"](.+?)[>"]', content):
            includes.append(match.group(1))
        result["imports"] = [{"header": include} for include in includes]
        
        # Extract functions
        for match in re.finditer(r'(?:[\w:*&]+\s+)+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*(?:noexcept)?\s*(?:override)?\s*(?:final)?\s*(?:(?:=\s*0)?|{)', content):
            name = match.group(1)
            params = match.group(2).strip()
            result["functions"].append({
                "name": name,
                "params": params
            })
        
        # Extract classes
        for match in re.finditer(r'(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+(\w+))?', content):
            name = match.group(1)
            inherits = match.group(2)
            result["classes"].append({
                "name": name,
                "inherits": [inherits] if inherits else []
            })
    
    def _analyze_java(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract Java code structure"""
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Extract package
        package_match = re.search(r'package\s+([\w.]+);', content)
        if package_match:
            result["package"] = package_match.group(1)
        
        # Extract imports
        for match in re.finditer(r'import\s+(static\s+)?([\w.]+)(?:\.([\w]+|\*));', content):
            is_static = bool(match.group(1))
            package = match.group(2)
            class_name = match.group(3)
            result["imports"].append({
                "static": is_static,
                "package": package,
                "class": class_name
            })
        
        # Extract classes
        for match in re.finditer(r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?', content):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)
            implements_list = []
            if implements:
                implements_list = [i.strip() for i in implements.split(',')]
            
            result["classes"].append({
                "name": name,
                "extends": extends,
                "implements": implements_list
            })
        
        # Extract methods
        for match in re.finditer(r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:[\w<>[\],\s]+)\s+(\w+)\s*\(([^)]*)\)', content):
            name = match.group(1)
            params = match.group(2).strip()
            result["functions"].append({
                "name": name,
                "params": params
            })
    
    def _analyze_binary(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract information from binary files"""
        try:
            # Use file command to get basic info
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            result["binary_info"] = output.strip()
            
            # Try objdump for more detailed info
            if os.name != "nt":  # objdump not available on Windows
                try:
                    headers = subprocess.check_output(
                        ["objdump", "-f", file_path],
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    result["binary_headers"] = headers.strip()
                    
                    # Try to get symbol table
                    symbols = subprocess.check_output(
                        ["objdump", "-t", file_path],
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Parse functions from symbol table
                    for line in symbols.splitlines():
                        if " F " in line:  # Function symbol
                            parts = line.split()
                            if len(parts) >= 6:
                                result["functions"].append({"name": parts[-1]})
                except:
                    pass
        except:
            pass

class Decompiler:
    """Handles decompilation of binary files into readable code"""
    
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.decompiled_dir = os.path.join(work_dir, "decompiled")
        os.makedirs(self.decompiled_dir, exist_ok=True)
    
    def decompile_binary(self, file_path: str, 
                         strategies: List[DecompStrategy] = None) -> List[str]:
        """
        Decompile a binary file using multiple strategies
        
        Args:
            file_path: Path to binary file
            strategies: List of decompilation strategies to try
            
        Returns:
            List of paths to decompiled files
        """
        if strategies is None:
            strategies = [
                DecompStrategy.RADARE2,
                DecompStrategy.RETDEC
            ]
        
        # Create a unique directory for this binary
        file_hash = self._hash_file(file_path)
        binary_name = os.path.basename(file_path)
        output_dir = os.path.join(self.decompiled_dir, f"{binary_name}_{file_hash[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        
        decompiled_files = []
        
        # Try each strategy
        for strategy in strategies:
            try:
                result_file = self._decompile_with_strategy(file_path, strategy, output_dir)
                if result_file and os.path.exists(result_file):
                    decompiled_files.append(result_file)
                    logger.info(f"Successfully decompiled {file_path} using {strategy.value}")
            except Exception as e:
                logger.error(f"Failed to decompile {file_path} using {strategy.value}: {str(e)}")
        
        if not decompiled_files:
            logger.warning(f"All decompilation strategies failed for {file_path}")
        
        return decompiled_files
    
    def _hash_file(self, file_path: str) -> str:
        """Create a hash of file contents for unique identification"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _decompile_with_strategy(self, file_path: str, 
                                 strategy: DecompStrategy, 
                                 output_dir: str) -> Optional[str]:
        """
        Decompile binary using a specific strategy
        
        Args:
            file_path: Path to binary file
            strategy: Decompilation strategy
            output_dir: Directory to store output
            
        Returns:
            Path to decompiled file if successful, None otherwise
        """
        if strategy == DecompStrategy.RADARE2:
            return self._decompile_with_radare2(file_path, output_dir)
        elif strategy == DecompStrategy.RETDEC:
            return self._decompile_with_retdec(file_path, output_dir)
        elif strategy == DecompStrategy.GHIDRA:
            return self._decompile_with_ghidra(file_path, output_dir)
        else:
            logger.error(f"Unsupported decompilation strategy: {strategy.value}")
            return None
    
    def _decompile_with_radare2(self, file_path: str, output_dir: str) -> Optional[str]:
        """Decompile using radare2"""
        output_file = os.path.join(output_dir, "radare2_decompiled.c")
        
        # Create a radare2 script
        script_file = os.path.join(output_dir, "r2_script.txt")
        with open(script_file, 'w') as f:
            f.write("aaa\n")  # Analyze all
            f.write("s main\n")  # Seek to main
            f.write("pdf\n")  # Print disassembly function
            f.write("s sym.main\n")  # Alternative main symbol
            f.write("pdf\n")
            f.write("pdc\n")  # Print decompiled code
        
        try:
            # Run radare2 with the script
            output = subprocess.check_output(
                ["r2", "-q", "-i", script_file, file_path],
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            with open(output_file, 'w') as f:
                f.write("// Decompiled with radare2\n")
                f.write("// Command: r2 -q -i script.txt " + file_path + "\n\n")
                f.write(output)
            
            return output_file
        except Exception as e:
            logger.error(f"Radare2 decompilation failed: {str(e)}")
            return None
    
    def _decompile_with_retdec(self, file_path: str, output_dir: str) -> Optional[str]:
        """Decompile using RetDec"""
        output_file = os.path.join(output_dir, "retdec_decompiled.c")
        
        try:
            # Run RetDec
            subprocess.run(
                ["retdec-decompiler", file_path, "-o", output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_file
        except Exception as e:
            logger.error(f"RetDec decompilation failed: {str(e)}")
            return None
    
    def _decompile_with_ghidra(self, file_path: str, output_dir: str) -> Optional[str]:
        """Decompile using Ghidra (requires Ghidra installation)"""
        output_file = os.path.join(output_dir, "ghidra_decompiled.c")
        
        # This is a simplified version - actual Ghidra integration requires more setup
        try:
            ghidra_path = os.environ.get("GHIDRA_PATH", "/opt/ghidra")
            headless_path = os.path.join(ghidra_path, "support", "analyzeHeadless")
            
            if not os.path.exists(headless_path):
                logger.error(f"Ghidra headless analyzer not found at {headless_path}")
                return None
            
            project_dir = os.path.join(output_dir, "ghidra_project")
            os.makedirs(project_dir, exist_ok=True)
            
            # Run Ghidra headless analyzer
            subprocess.run(
                [
                    headless_path,
                    project_dir,
                    "UnravelProject",
                    "-import", file_path,
                    "-postScript", "DecompileScript.java",
                    "-scriptPath", "/path/to/scripts",
                    "-noanalysis"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Look for the decompiled file
            for root, _, files in os.walk(project_dir):
                for file in files:
                    if file.endswith(".c") or file.endswith(".cpp"):
                        found_file = os.path.join(root, file)
                        shutil.copy(found_file, output_file)
                        return output_file
            
            logger.error("Ghidra decompilation completed but no output file found")
            return None
        except Exception as e:
            logger.error(f"Ghidra decompilation failed: {str(e)}")
            return None

class SpecGenerator:
    """Generates software specifications from decompiled code"""
    
    def __init__(self, work_dir: str, llm_endpoint: str = None):
        self.work_dir = work_dir
        self.specs_dir = os.path.join(work_dir, "specs")
        os.makedirs(self.specs_dir, exist_ok=True)
        self.llm_endpoint = llm_endpoint
        self.token_counter = TokenCounter()
        self.file_analyzer = FileAnalyzer()
    
    def generate_specifications(self, decompiled_files: List[str]) -> List[str]:
        """
        Generate specifications from decompiled files
        
        Args:
            decompiled_files: List of paths to decompiled files
            
        Returns:
            List of paths to generated specification files
        """
        if not decompiled_files:
            logger.warning("No decompiled files provided")
            return []
        
        # Create a unique directory for specs
        timestamp = int(os.path.getmtime(decompiled_files[0]))
        spec_dir = os.path.join(self.specs_dir, f"spec_{timestamp}")
        os.makedirs(spec_dir, exist_ok=True)
        
        spec_files = []
        
        # Generate combined specification
        combined_spec_path = os.path.join(spec_dir, "combined_spec.md")
        with open(combined_spec_path, 'w') as spec_file:
            spec_file.write("# Software Specification\n\n")
            spec_file.write("This document contains specifications extracted from the decompiled software.\n\n")
            
            # Process each decompiled file
            for decompiled_file in decompiled_files:
                file_name = os.path.basename(decompiled_file)
                spec_file.write(f"## {file_name}\n\n")
                
                try:
                    # Analyze the decompiled file
                    analysis_result = self.file_analyzer.analyze_file(decompiled_file)
                    
                    # Write specification based on analysis
                    spec_file.write(f"### Overview\n\n")
                    
                    file_type = FileType(analysis_result["file_type"])
                    spec_file.write(f"- **File Type**: {file_type.value}\n")
                    spec_file.write(f"- **Size**: {analysis_result['file_size']} bytes\n\n")
                    
                    # Functions
                    if analysis_result["functions"]:
                        spec_file.write("### Functions\n\n")
                        for func in analysis_result["functions"]:
                            spec_file.write(f"- `{func['name']}`\n")
                            if "params" in func:
                                spec_file.write(f"  - Parameters: `{func['params']}`\n")
                            if "return_type" in func and func["return_type"]:
                                spec_file.write(f"  - Returns: `{func['return_type']}`\n")
                        spec_file.write("\n")
                    
                    # Classes
                    if analysis_result["classes"]:
                        spec_file.write("### Classes\n\n")
                        for cls in analysis_result["classes"]:
                            spec_file.write(f"- `{cls['name']}`\n")
                            if "inherits" in cls and cls["inherits"]:
                                inherits_str = ", ".join(cls["inherits"])
                                spec_file.write(f"  - Inherits: `{inherits_str}`\n")
                        spec_file.write("\n")
                    
                    # Imports/Dependencies
                    if analysis_result["imports"]:
                        spec_file.write("### Dependencies\n\n")
                        for imp in analysis_result["imports"]:
                            if "module" in imp:
                                spec_file.write(f"- `{imp['module']}`")
                                if "name" in imp:
                                    spec_file.write(f" → `{imp['name']}`")
                                spec_file.write("\n")
                        spec_file.write("\n")
                    
                    # If this is a binary analysis, include additional info
                    if "binary_info" in analysis_result:
                        spec_file.write("### Binary Information\n\n")
                        spec_file.write(f"```\n{analysis_result['binary_info']}\n```\n\n")
                    
                    # Add raw file content for small text files
                    if file_type != FileType.BINARY and analysis_result["file_size"] < 10000:
                        with open(decompiled_file, 'r', errors='ignore') as f:
                            content = f.read()
                            spec_file.write("### Source Code\n\n")
                            spec_file.write(f"```{file_type.value}\n{content}\n```\n\n")
                    
                except Exception as e:
                    logger.error(f"Error processing {decompiled_file}: {str(e)}")
                    spec_file.write(f"Error processing file: {str(e)}\n\n")
        
        spec_files.append(combined_spec_path)
        
        # Generate specialized specs for different aspects
        self._generate_specialized_specs(decompiled_files, spec_dir, spec_files)
        
        # Generate prompts for LLM-based analysis
        self._generate_llm_prompts(decompiled_files, spec_dir, spec_files)
        
        return spec_files
    
    def _generate_specialized_specs(self, decompiled_files: List[str], 
                                   spec_dir: str, spec_files: List[str]) -> None:
        """Generate specialized specifications for different aspects"""
        # Data structures spec
        data_structures_path = os.path.join(spec_dir, "data_structures.md")
        with open(data_structures_path, 'w') as f:
            f.write("# Data Structures\n\n")
            f.write("This document describes the key data structures identified in the software.\n\n")
            
            for decompiled_file in decompiled_files:
                file_type = self.file_analyzer.detect_file_type(decompiled_file)
                if file_type in [FileType.C, FileType.CPP, FileType.JAVA, FileType.CSHARP]:
                    self._extract_data_structures(decompiled_file, f)
        
        spec_files.append(data_structures_path)
        
        # API documentation
        api_path = os.path.join(spec_dir, "api_documentation.md")
        with open(api_path, 'w') as f:
            f.write("# API Documentation\n\n")
            f.write("This document describes the public API of the software.\n\n")
            
            for decompiled_file in decompiled_files:
                self._extract_api_documentation(decompiled_file, f)
        
        spec_files.append(api_path)
        
        # Algorithm analysis
        algorithm_path = os.path.join(spec_dir, "algorithms.md")
        with open(algorithm_path, 'w') as f:
            f.write("# Algorithms\n\n")
            f.write("This document describes the key algorithms identified in the software.\n\n")
            
            for decompiled_file in decompiled_files:
                self._extract_algorithms(decompiled_file, f)
        
        spec_files.append(algorithm_path)
    
    def _extract_data_structures(self, file_path: str, outfile) -> None:
        """Extract data structures from a file"""
        file_name = os.path.basename(file_path)
        outfile.write(f"## Data Structures in {file_name}\n\n")
        
        file_type = self.file_analyzer.detect_file_type(file_path)
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Extract structs and classes
            if file_type in [FileType.C, FileType.CPP]:
                # Match struct and class definitions
                struct_pattern = r'(?:typedef\s+)?struct\s+(\w+)\s*{([^}]*)}'
                class_pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+\w+)?\s*{([^}]*)}'
                
                for pattern, kind in [(struct_pattern, "struct"), (class_pattern, "class")]:
                    for match in re.finditer(pattern, content, re.DOTALL):
                        name = match.group(1)
                        body = match.group(2)
                        
                        outfile.write(f"### {kind} `{name}`\n\n")
                        outfile.write("| Field | Type | Description |\n")
                        outfile.write("|-------|------|-------------|\n")
                        
                        # Extract fields
                        field_pattern = r'(\b\w+(?:\s+\w+)*\b(?:\s*\*+)?)\s+(\w+)(?:\s*\[\s*\w*\s*\])?;'
                        for field_match in re.finditer(field_pattern, body):
                            field_type = field_match.group(1).strip()
                            field_name = field_match.group(2).strip()
                            outfile.write(f"| {field_name} | {field_type} | |\n")
                        
                        outfile.write("\n")
            
            elif file_type == FileType.JAVA:
                # Match class definitions with fields
                class_pattern = r'(?:public|private|protected)?\s*class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*{([^}]*)}'
                
                for match in re.finditer(class_pattern, content, re.DOTALL):
                    name = match.group(1)
                    body = match.group(2)
                    
                    outfile.write(f"### class `{name}`\n\n")
                    outfile.write("| Field | Type | Visibility | Description |\n")
                    outfile.write("|-------|------|------------|-------------|\n")
                    
                    # Extract fields
                    field_pattern = r'(public|private|protected)\s+(\b\w+(?:<[^>]*>)?)\s+(\w+)(?:\s*=\s*[^;]*)?;'
                    for field_match in re.finditer(field_pattern, body):
                        visibility = field_match.group(1).strip()
                        field_type = field_match.group(2).strip()
                        field_name = field_match.group(3).strip()
                        outfile.write(f"| {field_name} | {field_type} | {visibility} | |\n")
                    
                    outfile.write("\n")
        
        except Exception as e:
            outfile.write(f"Error extracting data structures: {str(e)}\n\n")
    
    def _extract_api_documentation(self, file_path: str, outfile) -> None:
        """Extract API documentation from a file"""
        file_name = os.path.basename(file_path)
        outfile.write(f"## API in {file_name}\n\n")
        
        file_type = self.file_analyzer.detect_file_type(file_path)
        
        try:
            analysis = self.file_analyzer.analyze_file(file_path)
            
            # Extract public functions and methods
            if "functions" in analysis and analysis["functions"]:
                outfile.write("### Functions/Methods\n\n")
                
                for func in analysis["functions"]:
                    outfile.write(f"#### `{func['name']}`\n\n")
                    
                    if "params" in func:
                        outfile.write(f"**Parameters:** `{func['params']}`\n\n")
                    
                    if "return_type" in func and func["return_type"]:
                        outfile.write(f"**Returns:** `{func['return_type']}`\n\n")
                    
                    outfile.write("**Description:** \n\n")
                    outfile.write("*No description available*\n\n")
            
            # Extract exports for JavaScript/TypeScript
            if "exports" in analysis and analysis["exports"]:
                outfile.write("### Exports\n\n")
                
                for exp in analysis["exports"]:
                    outfile.write(f"- `{exp['name']}`\n")
                
                outfile.write("\n")
            
        except Exception as e:
            outfile.write(f"Error extracting API documentation: {str(e)}\n\n")
    
    def _extract_algorithms(self, file_path: str, outfile) -> None:
        """Extract and identify algorithms from a file"""
        file_name = os.path.basename(file_path)
        outfile.write(f"## Algorithms in {file_name}\n\n")
        
        # This is a complex task that often requires understanding of code semantics
        # For simple detection, we'll look for algorithm indicators in function names and comments
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Look for common algorithm names in function names
            algorithm_indicators = [
                "sort", "search", "find", "traverse", "encode", "decode", "encrypt", "decrypt",
                "compress", "decompress", "hash", "parse", "serialize", "deserialize", 
                "graph", "tree", "heap", "queue", "stack", "list", "map", "reduce", "filter"
            ]
            
            found_algorithms = set()
            
            # Check function names
            for indicator in algorithm_indicators:
                for match in re.finditer(fr'\b(\w*{indicator}\w*)\b', content, re.IGNORECASE):
                    found_algorithms.add(match.group(1))
            
            if found_algorithms:
                outfile.write("### Detected Algorithm Functions\n\n")
                for algo in sorted(found_algorithms):
                    outfile.write(f"- `{algo}`\n")
                outfile.write("\n")
            else:
                outfile.write("No specific algorithms detected.\n\n")
            
        except Exception as e:
            outfile.write(f"Error extracting algorithms: {str(e)}\n\n")
    
    def _generate_llm_prompts(self, decompiled_files: List[str], 
                              spec_dir: str, spec_files: List[str]) -> None:
        """Generate prompts for LLM-based analysis"""
        # Generate a prompt for each decompiled file
        for decompiled_file in decompiled_files:
            file_name = os.path.basename(decompiled_file)
            prompt_file = os.path.join(spec_dir, f"{file_name}.prompt.txt")
            
            try:
                with open(decompiled_file, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Skip large files
                if len(content) > 100000:  # ~100KB
                    logger.info(f"Skipping large file {file_name} for LLM prompt")
                    continue
                
                # Create a prompt for the file
                with open(prompt_file, 'w') as f:
                    f.write(f"""Analyze the following code and provide a comprehensive specification:
