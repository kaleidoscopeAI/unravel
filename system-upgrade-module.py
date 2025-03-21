#!/usr/bin/env python3
"""
Kaleidoscope AI - System Upgrade Module
=======================================
Automated system for upgrading and modernizing outdated codebases.
Preserves functionality while enhancing architecture, security, and performance.
"""

import os
import sys
import re
import ast
import json
import shutil
import tempfile
import subprocess
import importlib
import logging
import zipfile
import tarfile
import uuid
import hashlib
import datetime
import docker
import multiprocessing
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_upgrade.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    CSHARP = auto()
    CPP = auto()
    RUBY = auto()
    PHP = auto()
    GO = auto()
    RUST = auto()
    SWIFT = auto()
    KOTLIN = auto()
    UNKNOWN = auto()

class SystemType(Enum):
    """Types of systems to upgrade"""
    WEB_APP = auto()
    DESKTOP_APP = auto()
    MOBILE_APP = auto()
    API = auto()
    CLI = auto()
    LIBRARY = auto()
    FRAMEWORK = auto()
    DATABASE = auto()
    UNKNOWN = auto()

class UpgradeStrategy(Enum):
    """Strategies for system upgrades"""
    IN_PLACE = auto()  # Modify existing codebase
    INCREMENTAL = auto()  # Upgrade component by component
    FULL_REWRITE = auto()  # Complete rewrite with same language
    LANGUAGE_MIGRATION = auto()  # Rewrite in different language
    WRAPPER = auto()  # Create wrapper around existing system

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version: str
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    is_vulnerable: bool = False
    vulnerability_details: Optional[str] = None
    is_outdated: bool = False
    upgrade_path: Optional[str] = None
    is_deprecated: bool = False
    alternatives: List[str] = field(default_factory=list)

@dataclass
class CodeFile:
    """Information about a code file"""
    path: str
    language: LanguageType
    content: str
    ast: Optional[Any] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[DependencyInfo] = field(default_factory.list)
    vulnerabilities: List[str] = field(default_factory=list)
    outdated_patterns: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    is_test: bool = False

@dataclass
class SystemInfo:
    """Information about the system to upgrade"""
    root_path: str
    system_type: SystemType
    primary_language: LanguageType
    other_languages: List[LanguageType] = field(default_factory=list)
    files: Dict[str, CodeFile] = field(default_factory=dict)
    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    database_info: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    dependencies_graph: Optional[nx.DiGraph] = None
    file_count: int = 0
    code_size: int = 0  # In bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "root_path": self.root_path,
            "system_type": self.system_type.name,
            "primary_language": self.primary_language.name,
            "other_languages": [lang.name for lang in self.other_languages],
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "database_info": self.database_info,
            "api_endpoints": self.api_endpoints,
            "vulnerabilities": self.vulnerabilities,
            "file_count": self.file_count,
            "code_size": self.code_size,
            "dependencies": {k: {"name": v.name, "version": v.version} for k, v in self.dependencies.items()},
        }
        return result

@dataclass
class UpgradeConfig:
    """Configuration for the upgrade process"""
    target_language: LanguageType
    strategy: UpgradeStrategy
    preserve_functionality: bool = True
    update_dependencies: bool = True
    fix_vulnerabilities: bool = True
    improve_performance: bool = True
    add_tests: bool = True
    modernize_architecture: bool = True
    refactor_code: bool = True
    target_frameworks: List[str] = field(default_factory=list)
    excluded_paths: List[str] = field(default_factory=list)
    keep_original: bool = True
    max_parallel_processes: int = multiprocessing.cpu_count()
    timeout_seconds: int = 3600  # 1 hour

@dataclass
class UpgradeResult:
    """Results of the upgrade process"""
    success: bool
    output_path: str
    strategy_used: UpgradeStrategy
    upgraded_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    time_taken_seconds: float = 0.0
    size_difference: int = 0  # Difference in bytes
    applied_transformations: List[str] = field(default_factory=list)
    license_path: Optional[str] = None

class LanguageDetector:
    """Detects programming languages from file content and extensions"""
    
    def __init__(self):
        """Initialize language detector"""
        self.extension_map = {
            ".py": LanguageType.PYTHON,
            ".js": LanguageType.JAVASCRIPT,
            ".jsx": LanguageType.JAVASCRIPT,
            ".ts": LanguageType.TYPESCRIPT,
            ".tsx": LanguageType.TYPESCRIPT,
            ".java": LanguageType.JAVA,
            ".cs": LanguageType.CSHARP,
            ".cpp": LanguageType.CPP,
            ".cc": LanguageType.CPP,
            ".cxx": LanguageType.CPP,
            ".c": LanguageType.CPP,
            ".h": LanguageType.CPP,
            ".hpp": LanguageType.CPP,
            ".rb": LanguageType.RUBY,
            ".php": LanguageType.PHP,
            ".go": LanguageType.GO,
            ".rs": LanguageType.RUST,
            ".swift": LanguageType.SWIFT,
            ".kt": LanguageType.KOTLIN
        }
        
        self.shebang_patterns = {
            r"^\s*#!.*python": LanguageType.PYTHON,
            r"^\s*#!.*node": LanguageType.JAVASCRIPT,
            r"^\s*#!.*ruby": LanguageType.RUBY,
            r"^\s*#!.*php": LanguageType.PHP
        }
        
        self.content_patterns = {
            r"import\s+[a-zA-Z0-9_]+|from\s+[a-zA-Z0-9_\.]+\s+import": LanguageType.PYTHON,
            r"require\s*\(\s*['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*\)|import\s+[a-zA-Z0-9_]+\s+from": LanguageType.JAVASCRIPT,
            r"import\s+{\s*[a-zA-Z0-9_,\s]+\s*}\s+from|interface\s+[a-zA-Z0-9_]+": LanguageType.TYPESCRIPT,
            r"public\s+class|import\s+java\.": LanguageType.JAVA,
            r"namespace\s+[a-zA-Z0-9_\.]+|using\s+[a-zA-Z0-9_\.]+;": LanguageType.CSHARP,
            r"#include\s*<[a-zA-Z0-9_\.]+>|#include\s*\"[a-zA-Z0-9_\.]+\"": LanguageType.CPP,
            r"require\s+['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*|def\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUBY,
            r"<\?php|namespace\s+[a-zA-Z0-9_\\]+;": LanguageType.PHP,
            r"package\s+[a-zA-Z0-9_]+|func\s+[a-zA-Z0-9_]+\s*\(": LanguageType.GO,
            r"use\s+[a-zA-Z0-9_:]+|fn\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUST,
            r"import\s+[a-zA-Z0-9_\.]+|class\s+[a-zA-Z0-9_]+\s*:": LanguageType.SWIFT,
            r"package\s+[a-zA-Z0-9_\.]+|fun\s+[a-zA-Z0-9_]+\s*\(": LanguageType.KOTLIN
        }
    
    def detect_language(self, file_path: str, content: Optional[str] = None) -> LanguageType:
        """
        Detect the programming language of a file
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Detected language type
        """
        # Try by extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # If no content provided, try to read it
        if content is None:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {str(e)}")
                return LanguageType.UNKNOWN
        
        # Try by shebang
        for pattern, lang in self.shebang_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        # Try by content patterns
        for pattern, lang in self.content_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        return LanguageType.UNKNOWN

class SystemAnalyzer:
    """Analyzes a system to gather information needed for upgrading"""
    
    def __init__(self):
        """Initialize system analyzer"""
        self.language_detector = LanguageDetector()
        self.excluded_dirs = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__", 
            "venv", "env", ".env", ".venv", "dist", "build"
        }
        self.excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"
        }
    
    def analyze_system(self, path: str) -> SystemInfo:
        """
        Analyze a system to gather information
        
        Args:
            path: Path to the system root directory
            
        Returns:
            System information
        """
        logger.info(f"Analyzing system at {path}")
        
        # Initialize system info
        system_info = SystemInfo(
            root_path=path,
            system_type=SystemType.UNKNOWN,
            primary_language=LanguageType.UNKNOWN
        )
        
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        
        # Count languages for later determining primary language
        language_counts = {}
        
        # Walk through the directory tree
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path, topdown=True):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Process each file
            for file in files:
                if file in self.excluded_files:
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, path)
                
                # Skip binary files and large files
                if self._is_binary_file(file_path) or os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                    continue
                
                try:
                    # Read file content
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                    
                    # Detect language
                    language = self.language_detector.detect_language(file_path, content)
                    
                    # Update language counts
                    if language != LanguageType.UNKNOWN:
                        language_counts[language] = language_counts.get(language, 0) + 1
                    
                    # Create code file info
                    code_file = CodeFile(
                        path=relative_path,
                        language=language,
                        content=content
                    )
                    
                    # Extract imports and other information based on language
                    self._extract_file_info(code_file)
                    
                    # Add to system info
                    system_info.files[relative_path] = code_file
                    
                    # Update total size
                    file_size = len(content.encode('utf-8'))
                    total_size += file_size
                    file_count += 1
                    
                    # Check for special files
                    file_lower = file.lower()
                    if any(name in file_lower for name in ["readme", "license", "dockerfile", "docker-compose"]):
                        # Could add special handling here
                        pass
                    
                    # Identify potential entry points
                    if self._is_entry_point(file_path, relative_path, language):
                        system_info.entry_points.append(relative_path)
                    
                    # Identify configuration files
                    if self._is_config_file(file_path, relative_path):
                        system_info.config_files.append(relative_path)
                
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
        
        # Set primary language and other languages
        if language_counts:
            primary_language = max(language_counts.items(), key=lambda x: x[1])[0]
            system_info.primary_language = primary_language
            system_info.other_languages = [lang for lang in language_counts.keys() if lang != primary_language]
        
        # Determine system type
        system_info.system_type = self._determine_system_type(system_info)
        
        # Update file count and code size
        system_info.file_count = file_count
        system_info.code_size = total_size
        
        # Build dependency graph
        system_info.dependencies_graph = self._build_dependency_graph(system_info)
        
        # Analyze dependencies
        self._analyze_dependencies(system_info)
        
        # Identify API endpoints
        self._identify_api_endpoints(system_info)
        
        # Check for vulnerabilities
        self._check_vulnerabilities(system_info)
        
        # Identify database connections
        self._identify_database_connections(system_info)
        
        logger.info(f"System analysis complete: {system_info.primary_language.name}, {system_info.system_type.name}, {file_count} files")
        
        return system_info
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary"""
        try:
            with open(file_path, 'r') as f:
                f.read(1024)
            return False
        except UnicodeDecodeError:
            return True
    
    def _extract_file_info(self, code_file: CodeFile) -> None:
        """Extract imports and other information from file"""
        language = code_file.language
        content = code_file.content
        
        if language == LanguageType.PYTHON:
            self._extract_python_imports(code_file)
        elif language == LanguageType.JAVASCRIPT:
            self._extract_javascript_imports(code_file)
        elif language == LanguageType.TYPESCRIPT:
            self._extract_typescript_imports(code_file)
        elif language == LanguageType.JAVA:
            self._extract_java_imports(code_file)
    
    def _extract_python_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Python file"""
        try:
            tree = ast.parse(code_file.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        code_file.imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        for name in node.names:
                            code_file.imports.append(f"{module_name}.{name.name}")
        except SyntaxError:
            # Fall back to regex for invalid Python
            for match in re.finditer(r'^\s*(?:import|from)\s+([\w\.]+)', code_file.content, re.MULTILINE):
                code_file.imports.append(match.group(1))
    
    def _extract_javascript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from JavaScript file"""
        # ES6 imports
        for match in re.finditer(r'import\s+(?:{\s*([\w\s,]+)\s*}|(\w+))\s+from\s+[\'"]([^\'"]*)[\'"]\s*;?', code_file.content):
            if match.group(1):  # Named imports
                for name in match.group(1).split(','):
                    code_file.imports.append(name.strip())
            elif match.group(2):  # Default import
                code_file.imports.append(match.group(2))
        
        # CommonJS requires
        for match in re.finditer(r'(?:const|let|var)\s+([\w{}:\s,]+)\s*=\s*require\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)\s*;?', code_file.content):
            code_file.imports.append(match.group(2))
    
    def _extract_typescript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from TypeScript file"""
        # TypeScript imports are similar to JavaScript
        self._extract_javascript_imports(code_file)
    
    def _extract_java_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Java file"""
        for match in re.finditer(r'import\s+([\w\.]+)(?:\.\*)?;', code_file.content):
            code_file.imports.append(match.group(1))
    
    def _is_entry_point(self, file_path: str, relative_path: str, language: LanguageType) -> bool:
        """Identify if a file is an entry point"""
        file_name = os.path.basename(file_path).lower()
        
        # Common entry point patterns
        if language == LanguageType.PYTHON:
            return file_name in ["main.py", "app.py", "manage.py", "run.py"] or "if __name__ == '__main__'" in open(file_path, 'r', errors='ignore').read()
        elif language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
            return file_name in ["index.js", "main.js", "app.js", "server.js", "index.ts", "main.ts", "app.ts", "server.ts"]
        elif language == LanguageType.JAVA:
            return "public static void main(" in open(file_path, 'r', errors='ignore').read()
        elif language == LanguageType.CSHARP:
            return "static void Main(" in open(file_path, 'r', errors='ignore').read() or "Program.cs" in file_path
        
        return False
    
    def _is_config_file(self, file_path: str, relative_path: str) -> bool:
        """Identify if a file is a configuration file"""
        file_name = os.path.basename(file_path).lower()
        ext = os.path.splitext(file_name)[1].lower()
        
        config_patterns = [
            "config", "settings", ".env", ".ini", ".yml", ".yaml", ".json", ".xml", ".toml",
            "package.json", "composer.json", "pyproject.toml", "requirements.txt", "Gemfile",
            ".gitignore", "Dockerfile", "docker-compose"
        ]
        
        return any(pattern in file_name for pattern in config_patterns)
    
    def _determine_system_type(self, system_info: SystemInfo) -> SystemType:
        """Determine the type of system"""
        files = system_info.files
        
        # Web app indicators
        web_indicators = [
            "index.html", "app.js", "webpack.config.js", "package.json", 
            "views", "templates", "public", "static", "assets"
        ]
        
        # API indicators
        api_indicators = [
            "routes", "controllers", "endpoints", "api", "rest", "graphql", 
            "swagger", "openapi"
        ]
        
        # Desktop app indicators
        desktop_indicators = [
            "electron", "qt", "gtk", "wxwidgets", "window", "mainwindow", "form"
        ]
        
        # Count indicators
        web_score = sum(1 for f in files if any(ind in f.lower() for ind in web_indicators))
        api_score = sum(1 for f in files if any(ind in f.lower() for ind in api_indicators))
        desktop_score = sum(1 for f in files if any(ind in f.lower() for ind in desktop_indicators))
        
        # Additional checks for specific files and content
        for f_path, code_file in files.items():
            # Check file content for indicators
            content = code_file.content.lower()
            
            if "<!doctype html>" in content or "<html" in content:
                web_score += 1
            
            if "api" in content and ("endpoint" in content or "route" in content):
                api_score += 1
            
            if "window" in content and ("gui" in content or "interface" in content):
                desktop_score += 1
        
        # Determine type based on scores
        max_score = max(web_score, api_score, desktop_score)
        
        if max_score == 0:
            # Check if it's a library/framework
            if any("setup.py" in f or "package.json" in f for f in files):
                return SystemType.LIBRARY
            return SystemType.UNKNOWN
        
        if max_score == web_score:
            return SystemType.WEB_APP
        elif max_score == api_score:
            return SystemType.API
        elif max_score == desktop_score:
            return SystemType.DESKTOP_APP
        
        return SystemType.UNKNOWN
    
    def _build_dependency_graph(self, system_info: SystemInfo) -> nx.DiGraph:
        """Build a dependency graph of files"""
        G = nx.DiGraph()
        
        # Add all files as nodes
        for file_path in system_info.files:
            G.add_node(file_path)
        
        # Add edges based on imports
        for file_path, code_file in system_info.files.items():
            for imported in code_file.imports:
                # Try to find the corresponding file
                for other_path, other_file in system_info.files.items():
                    if self._file_provides_import(other_file, imported):
                        G.add_edge(file_path, other_path)
        
        return G
    
    def _file_provides_import(self, code_file: CodeFile, import_name: str) -> bool:
        """Check if a file provides the given import"""
        # Very simple check for now
        file_basename = os.path.splitext(os.path.basename(code_file.path))[0]
        return file_basename == import_name or import_name.endswith(f".{file_basename}")
    
    def _analyze_dependencies(self, system_info: SystemInfo) -> None:
        """Analyze external dependencies"""
        # Extract dependencies from common dependency files
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            
            if "requirements.txt" in file_path:
                self._extract_python_dependencies(system_info, full_path)
            elif "package.json" in file_path:
                self._extract_npm_dependencies(system_info, full_path)
            elif "composer.json" in file_path:
                self._extract_composer_dependencies(system_info, full_path)
            elif "gemfile" in file_path.lower():
                self._extract_ruby_dependencies(system_info, full_path)
            elif "build.gradle" in file_path or "pom.xml" in file_path:
                self._extract_java_dependencies(system_info, full_path)
    
    def _extract_python_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Python dependencies from requirements.txt"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse dependency
                    parts = re.split(r'[=<>]', line, 1)
                    name = parts[0].strip()
                    version = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Add to dependencies
                    system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_npm_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract NPM dependencies from package.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['dependencies', 'devDependencies']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_composer_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract PHP dependencies from composer.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['require', 'require-dev']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_ruby_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Ruby dependencies from Gemfile"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Match gem declarations
                    match = re.match(r'gem\s+[\'"]([^\'"]+)[\'"](?:,\s*[\'"]([^\'"]+)[\'"])?', line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) or ""
                        
                        # Add to dependencies
                        system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_java_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Java dependencies from build.gradle or pom.xml"""
        try:
            if file_path.endswith("build.gradle"):
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Extract dependencies from build.gradle using regex
                    for match in re.finditer(r'(?:compile|implementation|api)\s+[\'"]([^:]+):([^:]+):([^\'"]+)[\'"]', content):
                        group = match.group(1)
                        name = match.group(2)
                        version = match.group(3)
                        
                        full_name = f"{group}:{name}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
            elif file_path.endswith("pom.xml"):
                # Simple regex-based extraction for POMs
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract dependencies from pom.xml
                    dependencies = re.findall(r'<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>', content)
                    for group, artifact, version in dependencies:
                        full_name = f"{group}:{artifact}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _identify_api_endpoints(self, system_info: SystemInfo) -> None:
        """Identify API endpoints"""
        # Only process if it's an API or web app
        if system_info.system_type not in [SystemType.API, SystemType.WEB_APP]:
            return
        
        endpoints = []
        
        for file_path, code_file in system_info.files.items():
            # Check based on language
            if code_file.language == LanguageType.PYTHON:
                self._extract_python_endpoints(code_file, endpoints)
            elif code_file.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                self._extract_js_endpoints(code_file, endpoints)
            elif code_file.language == LanguageType.JAVA:
                self._extract_java_endpoints(code_file, endpoints)
            # Add more languages as needed
        
        system_info.api_endpoints = endpoints
    
    def _extract_python_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Python file"""
        content = code_file.content
        
        # Flask endpoints
        for match in re.finditer(r'@(?:app|blueprint)\.route\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # Django URLs
        for match in re.finditer(r'path\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # FastAPI endpoints
        for match in re.finditer(r'@(?:app|router)\.(?:get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_js_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from JavaScript/TypeScript file"""
        content = code_file.content
        
        # Express.js endpoints
        for method in ['get', 'post', 'put', 'delete', 'patch']:
            for match in re.finditer(rf'(?:app|router)\.{method}\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
                endpoints.append(match.group(1))
        
        # Generic route definitions
        for match in re.finditer(r'route\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_java_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Java file"""
        content = code_file.content
        
        # Spring endpoints
        for match in re.finditer(r'@RequestMapping\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # JAX-RS endpoints
        for match in re.finditer(r'@Path\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _check_vulnerabilities(self, system_info: SystemInfo) -> None:
        """Check for known vulnerabilities"""
        vulnerabilities = []
        
        # In a real implementation, this would use a security database or API
        # For now, we'll look for some common vulnerability patterns
        security_patterns = {
            # SQL Injection
            r'(?:SELECT|INSERT|UPDATE|DELETE).*\+\s*["\']': "Potential SQL Injection",
            # XSS
            r'(?:innerHTML|document\.write)\s*\(': "Potential XSS vulnerability",
            # Hardcoded credentials
            r'(?:password|secret|key|token)\s*=\s*["\'][^"\']+["\']': "Hardcoded credentials",
            # Command injection
            r'(?:exec|spawn|system)\s*\(': "Potential command injection",
            # Insecure file operations
            r'eval\s*\(': "Insecure eval() usage"
        }
        
        for file_path, code_file in system_info.files.items():
            content = code_file.content
            
            for pattern, issue in security_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    vulnerability = f"{issue} in {file_path}"
                    vulnerabilities.append(vulnerability)
                    code_file.vulnerabilities.append(issue)
        
        # Also check for outdated dependencies with known vulnerabilities
        # In a real implementation, this would check against a vulnerability database
        
        system_info.vulnerabilities = vulnerabilities
    
    def _identify_database_connections(self, system_info: SystemInfo) -> None:
        """Identify database connections"""
        db_info = {}
        
        # Database patterns to look for
        db_patterns = {
            "mysql": r'mysql|mysqli|pdo_mysql',
            "postgres": r'postgres|pg_connect|pdo_pgsql',
            "sqlite": r'sqlite|pdo_sqlite',
            "mongodb": r'mongodb|mongo_connect',
            "oracle": r'oracle|oci_connect',
            "sqlserver": r'sqlserver|mssql|pdo_sqlsrv'
        }
        
        # Check configuration files first
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            try:
                with open(full_path, 'r') as f:
                    content = f.read().lower()
                    
                    # Look for connection strings
                    for db_type, pattern in db_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            if db_type not in db_info:
                                db_info[db_type] = []
                            db_info[db_type].append(file_path)
                            
                            # Look for connection parameters
                            for param in ["host", "port", "database", "dbname", "user", "username", "pass", "password"]:
                                matches = re.finditer(rf'{param}\s*[=:]\s*[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
                                for match in matches:
                                    if "connection_params" not in db_info:
                                        db_info["connection_params"] = {}
                                    db_info["connection_params"][param] = match.group(1)
            except Exception as e:
                logger.warning(f"Error checking database info in {file_path}: {str(e)}")
        
        # Also check code files
        for file_path, code_file in system_info.files.items():
            content = code_file.content.lower()
            
            # Look for database imports and connection code
            for db_type, pattern in db_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    if db_type not in db_info:
                        db_info[db_type] = []
                    db_info[db_type].append(file_path)
        
        system_info.database_info = db_info

class CodeTransformer(ABC):
    """Base class for code transformers"""
    
    @abstractmethod
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        pass
    
    @abstractmethod
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """
        Transform the code
        
        Args:
            code_file: Code file to transform
            system_info: System information
            
        Returns:
            Tuple of (transformed code, list of applied transformations)
        """
        pass

class PythonModernizer(CodeTransformer):
    """Modernizes Python code"""
    
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        return code_file.language == LanguageType.PYTHON
    
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """Transform Python code to modern standards"""
        content = code_file.content
        transformations = []
        
        # Add type hints
        content, type_transforms = self._add_type_hints(content)
        if type_transforms:
            transformations.append("Added type hints")
        
        # Convert to f-strings
        content, fstring_count = self._convert_to_fstrings(content)
        if fstring_count > 0:
            transformations.append(f"Converted {fstring_count} string formats to f-strings")
        
        # Use modern Python features
        content, modern_transforms = self._modernize_python_features(content)
        transformations.extend(modern_transforms)
        
        # Update imports
        content, import_transforms = self._update_imports(content, system_info)
        transformations.extend(import_transforms)
        
        return content, transformations
    
    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add type hints to Python code"""
        # This would require more sophisticated parsing
        # For a simple example, we'll just add typing import
        if 'from typing import ' not in content and 'import typing' not in content:
            content = "from typing import List, Dict, Tuple, Optional, Any, Union\n" + content
            return content, ["Added typing imports"]
        return content, []
    
    def _convert_to_fstrings(self, content: str) -> Tuple[str, int]:
        """Convert old-style string formatting to f-strings"""
        # Convert .format() style
        pattern = r'([\'"].*?[\'"])\s*\.\s*format\s*\((.*?)\)'
        
        count = 0
        for match in re.finditer(pattern, content):
            old_str = match.group(0)
            string_content = match.group(1)[1:-1]  # Remove quotes
            format_args = match.group(2)
            
            # Simple conversion for basic cases
            if not format_args.strip():
                continue
                
            # Try to convert
            try:
                # If format args are simple like "var1, var2"
                if re.match(r'^[\w\s,]+#!/usr/bin/env python3
"""
Kaleidoscope AI - System Upgrade Module
=======================================
Automated system for upgrading and modernizing outdated codebases.
Preserves functionality while enhancing architecture, security, and performance.
"""

import os
import sys
import re
import ast
import json
import shutil
import tempfile
import subprocess
import importlib
import logging
import zipfile
import tarfile
import uuid
import hashlib
import datetime
import docker
import multiprocessing
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_upgrade.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    CSHARP = auto()
    CPP = auto()
    RUBY = auto()
    PHP = auto()
    GO = auto()
    RUST = auto()
    SWIFT = auto()
    KOTLIN = auto()
    UNKNOWN = auto()

class SystemType(Enum):
    """Types of systems to upgrade"""
    WEB_APP = auto()
    DESKTOP_APP = auto()
    MOBILE_APP = auto()
    API = auto()
    CLI = auto()
    LIBRARY = auto()
    FRAMEWORK = auto()
    DATABASE = auto()
    UNKNOWN = auto()

class UpgradeStrategy(Enum):
    """Strategies for system upgrades"""
    IN_PLACE = auto()  # Modify existing codebase
    INCREMENTAL = auto()  # Upgrade component by component
    FULL_REWRITE = auto()  # Complete rewrite with same language
    LANGUAGE_MIGRATION = auto()  # Rewrite in different language
    WRAPPER = auto()  # Create wrapper around existing system

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version: str
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    is_vulnerable: bool = False
    vulnerability_details: Optional[str] = None
    is_outdated: bool = False
    upgrade_path: Optional[str] = None
    is_deprecated: bool = False
    alternatives: List[str] = field(default_factory=list)

@dataclass
class CodeFile:
    """Information about a code file"""
    path: str
    language: LanguageType
    content: str
    ast: Optional[Any] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[DependencyInfo] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory.list)
    outdated_patterns: List[str] = field(default_factory.list)
    complexity_score: float = 0.0
    is_test: bool = False

@dataclass
class SystemInfo:
    """Information about the system to upgrade"""
    root_path: str
    system_type: SystemType
    primary_language: LanguageType
    other_languages: List[LanguageType] = field(default_factory.list)
    files: Dict[str, CodeFile] = field(default_factory.dict)
    dependencies: Dict[str, DependencyInfo] = field(default_factory.dict)
    entry_points: List[str] = field(default_factory.list)
    config_files: List[str] = field(default_factory.list)
    database_info: Dict[str, Any] = field(default_factory.dict)
    api_endpoints: List[str] = field(default_factory.list)
    vulnerabilities: List[str] = field(default_factory.list)
    dependencies_graph: Optional[nx.DiGraph] = None
    file_count: int = 0
    code_size: int = 0  # In bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "root_path": self.root_path,
            "system_type": self.system_type.name,
            "primary_language": self.primary_language.name,
            "other_languages": [lang.name for lang in self.other_languages],
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "database_info": self.database_info,
            "api_endpoints": self.api_endpoints,
            "vulnerabilities": self.vulnerabilities,
            "file_count": self.file_count,
            "code_size": self.code_size,
            "dependencies": {k: {"name": v.name, "version": v.version} for k, v in self.dependencies.items()},
        }
        return result

@dataclass
class UpgradeConfig:
    """Configuration for the upgrade process"""
    target_language: LanguageType
    strategy: UpgradeStrategy
    preserve_functionality: bool = True
    update_dependencies: bool = True
    fix_vulnerabilities: bool = True
    improve_performance: bool = True
    add_tests: bool = True
    modernize_architecture: bool = True
    refactor_code: bool = True
    target_frameworks: List[str] = field(default_factory.list)
    excluded_paths: List[str] = field(default_factory.list)
    keep_original: bool = True
    max_parallel_processes: int = multiprocessing.cpu_count()
    timeout_seconds: int = 3600  # 1 hour

@dataclass
class UpgradeResult:
    """Results of the upgrade process"""
    success: bool
    output_path: str
    strategy_used: UpgradeStrategy
    upgraded_files: List[str] = field(default_factory.list)
    errors: List[str] = field(default_factory.list)
    backup_path: Optional[str] = None
    time_taken_seconds: float = 0.0
    size_difference: int = 0  # Difference in bytes
    applied_transformations: List[str] = field(default_factory.list)
    license_path: Optional[str] = None

class LanguageDetector:
    """Detects programming languages from file content and extensions"""
    
    def __init__(self):
        """Initialize language detector"""
        self.extension_map = {
            ".py": LanguageType.PYTHON,
            ".js": LanguageType.JAVASCRIPT,
            ".jsx": LanguageType.JAVASCRIPT,
            ".ts": LanguageType.TYPESCRIPT,
            ".tsx": LanguageType.TYPESCRIPT,
            ".java": LanguageType.JAVA,
            ".cs": LanguageType.CSHARP,
            ".cpp": LanguageType.CPP,
            ".cc": LanguageType.CPP,
            ".cxx": LanguageType.CPP,
            ".c": LanguageType.CPP,
            ".h": LanguageType.CPP,
            ".hpp": LanguageType.CPP,
            ".rb": LanguageType.RUBY,
            ".php": LanguageType.PHP,
            ".go": LanguageType.GO,
            ".rs": LanguageType.RUST,
            ".swift": LanguageType.SWIFT,
            ".kt": LanguageType.KOTLIN
        }
        
        self.shebang_patterns = {
            r"^\s*#!.*python": LanguageType.PYTHON,
            r"^\s*#!.*node": LanguageType.JAVASCRIPT,
            r"^\s*#!.*ruby": LanguageType.RUBY,
            r"^\s*#!.*php": LanguageType.PHP
        }
        
        self.content_patterns = {
            r"import\s+[a-zA-Z0-9_]+|from\s+[a-zA-Z0-9_\.]+\s+import": LanguageType.PYTHON,
            r"require\s*\(\s*['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*\)|import\s+[a-zA-Z0-9_]+\s+from": LanguageType.JAVASCRIPT,
            r"import\s+{\s*[a-zA-Z0-9_,\s]+\s*}\s+from|interface\s+[a-zA-Z0-9_]+": LanguageType.TYPESCRIPT,
            r"public\s+class|import\s+java\.": LanguageType.JAVA,
            r"namespace\s+[a-zA-Z0-9_\.]+|using\s+[a-zA-Z0-9_\.]+;": LanguageType.CSHARP,
            r"#include\s*<[a-zA-Z0-9_\.]+>|#include\s*\"[a-zA-Z0-9_\.]+\"": LanguageType.CPP,
            r"require\s+['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*|def\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUBY,
            r"<\?php|namespace\s+[a-zA-Z0-9_\\]+;": LanguageType.PHP,
            r"package\s+[a-zA-Z0-9_]+|func\s+[a-zA-Z0-9_]+\s*\(": LanguageType.GO,
            r"use\s+[a-zA-Z0-9_:]+|fn\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUST,
            r"import\s+[a-zA-Z0-9_\.]+|class\s+[a-zA-Z0-9_]+\s*:": LanguageType.SWIFT,
            r"package\s+[a-zA-Z0-9_\.]+|fun\s+[a-zA-Z0-9_]+\s*\(": LanguageType.KOTLIN
        }
    
    def detect_language(self, file_path: str, content: Optional[str] = None) -> LanguageType:
        """
        Detect the programming language of a file
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Detected language type
        """
        # Try by extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # If no content provided, try to read it
        if content is None:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {str(e)}")
                return LanguageType.UNKNOWN
        
        # Try by shebang
        for pattern, lang in self.shebang_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        # Try by content patterns
        for pattern, lang in self.content_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        return LanguageType.UNKNOWN

class SystemAnalyzer:
    """Analyzes a system to gather information needed for upgrading"""
    
    def __init__(self):
        """Initialize system analyzer"""
        self.language_detector = LanguageDetector()
        self.excluded_dirs = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__", 
            "venv", "env", ".env", ".venv", "dist", "build"
        }
        self.excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"
        }
    
    def analyze_system(self, path: str) -> SystemInfo:
        """
        Analyze a system to gather information
        
        Args:
            path: Path to the system root directory
            
        Returns:
            System information
        """
        logger.info(f"Analyzing system at {path}")
        
        # Initialize system info
        system_info = SystemInfo(
            root_path=path,
            system_type=SystemType.UNKNOWN,
            primary_language=LanguageType.UNKNOWN
        )
        
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        
        # Count languages for later determining primary language
        language_counts = {}
        
        # Walk through the directory tree
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path, topdown=True):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Process each file
            for file in files:
                if file in self.excluded_files:
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, path)
                
                # Skip binary files and large files
                if self._is_binary_file(file_path) or os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                    continue
                
                try:
                    # Read file content
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                    
                    # Detect language
                    language = self.language_detector.detect_language(file_path, content)
                    
                    # Update language counts
                    if language != LanguageType.UNKNOWN:
                        language_counts[language] = language_counts.get(language, 0) + 1
                    
                    # Create code file info
                    code_file = CodeFile(
                        path=relative_path,
                        language=language,
                        content=content
                    )
                    
                    # Extract imports and other information based on language
                    self._extract_file_info(code_file)
                    
                    # Add to system info
                    system_info.files[relative_path] = code_file
                    
                    # Update total size
                    file_size = len(content.encode('utf-8'))
                    total_size += file_size
                    file_count += 1
                    
                    # Check for special files
                    file_lower = file.lower()
                    if any(name in file_lower for name in ["readme", "license", "dockerfile", "docker-compose"]):
                        # Could add special handling here
                        pass
                    
                    # Identify potential entry points
                    if self._is_entry_point(file_path, relative_path, language):
                        system_info.entry_points.append(relative_path)
                    
                    # Identify configuration files
                    if self._is_config_file(file_path, relative_path):
                        system_info.config_files.append(relative_path)
                
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
        
        # Set primary language and other languages
        if language_counts:
            primary_language = max(language_counts.items(), key=lambda x: x[1])[0]
            system_info.primary_language = primary_language
            system_info.other_languages = [lang for lang in language_counts.keys() if lang != primary_language]
        
        # Determine system type
        system_info.system_type = self._determine_system_type(system_info)
        
        # Update file count and code size
        system_info.file_count = file_count
        system_info.code_size = total_size
        
        # Build dependency graph
        system_info.dependencies_graph = self._build_dependency_graph(system_info)
        
        # Analyze dependencies
        self._analyze_dependencies(system_info)
        
        # Identify API endpoints
        self._identify_api_endpoints(system_info)
        
        # Check for vulnerabilities
        self._check_vulnerabilities(system_info)
        
        # Identify database connections
        self._identify_database_connections(system_info)
        
        logger.info(f"System analysis complete: {system_info.primary_language.name}, {system_info.system_type.name}, {file_count} files")
        
        return system_info
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary"""
        try:
            with open(file_path, 'r') as f:
                f.read(1024)
            return False
        except UnicodeDecodeError:
            return True
    
    def _extract_file_info(self, code_file: CodeFile) -> None:
        """Extract imports and other information from file"""
        language = code_file.language
        content = code_file.content
        
        if language == LanguageType.PYTHON:
            self._extract_python_imports(code_file)
        elif language == LanguageType.JAVASCRIPT:
            self._extract_javascript_imports(code_file)
        elif language == LanguageType.TYPESCRIPT:
            self._extract_typescript_imports(code_file)
        elif language == LanguageType.JAVA:
            self._extract_java_imports(code_file)
    
    def _extract_python_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Python file"""
        try:
            tree = ast.parse(code_file.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        code_file.imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        for name in node.names:
                            code_file.imports.append(f"{module_name}.{name.name}")
        except SyntaxError:
            # Fall back to regex for invalid Python
            for match in re.finditer(r'^\s*(?:import|from)\s+([\w\.]+)', code_file.content, re.MULTILINE):
                code_file.imports.append(match.group(1))
    
    def _extract_javascript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from JavaScript file"""
        # ES6 imports
        for match in re.finditer(r'import\s+(?:{\s*([\w\s,]+)\s*}|(\w+))\s+from\s+[\'"]([^\'"]*)[\'"]\s*;?', code_file.content):
            if match.group(1):  # Named imports
                for name in match.group(1).split(','):
                    code_file.imports.append(name.strip())
            elif match.group(2):  # Default import
                code_file.imports.append(match.group(2))
        
        # CommonJS requires
        for match in re.finditer(r'(?:const|let|var)\s+([\w{}:\s,]+)\s*=\s*require\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)\s*;?', code_file.content):
            code_file.imports.append(match.group(2))
    
    def _extract_typescript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from TypeScript file"""
        # TypeScript imports are similar to JavaScript
        self._extract_javascript_imports(code_file)
    
    def _extract_java_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Java file"""
        for match in re.finditer(r'import\s+([\w\.]+)(?:\.\*)?;', code_file.content):
            code_file.imports.append(match.group(1))
    
    def _is_entry_point(self, file_path: str, relative_path: str, language: LanguageType) -> bool:
        """Identify if a file is an entry point"""
        file_name = os.path.basename(file_path).lower()
        
        # Common entry point patterns
        if language == LanguageType.PYTHON:
            return file_name in ["main.py", "app.py", "manage.py", "run.py"] or "if __name__ == '__main__'" in open(file_path, 'r', errors='ignore').read()
        elif language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
            return file_name in ["index.js", "main.js", "app.js", "server.js", "index.ts", "main.ts", "app.ts", "server.ts"]
        elif language == LanguageType.JAVA:
            return "public static void main(" in open(file_path, 'r', errors='ignore').read()
        elif language == LanguageType.CSHARP:
            return "static void Main(" in open(file_path, 'r', errors='ignore').read() or "Program.cs" in file_path
        
        return False
    
    def _is_config_file(self, file_path: str, relative_path: str) -> bool:
        """Identify if a file is a configuration file"""
        file_name = os.path.basename(file_path).lower()
        ext = os.path.splitext(file_name)[1].lower()
        
        config_patterns = [
            "config", "settings", ".env", ".ini", ".yml", ".yaml", ".json", ".xml", ".toml",
            "package.json", "composer.json", "pyproject.toml", "requirements.txt", "Gemfile",
            ".gitignore", "Dockerfile", "docker-compose"
        ]
        
        return any(pattern in file_name for pattern in config_patterns)
    
    def _determine_system_type(self, system_info: SystemInfo) -> SystemType:
        """Determine the type of system"""
        files = system_info.files
        
        # Web app indicators
        web_indicators = [
            "index.html", "app.js", "webpack.config.js", "package.json", 
            "views", "templates", "public", "static", "assets"
        ]
        
        # API indicators
        api_indicators = [
            "routes", "controllers", "endpoints", "api", "rest", "graphql", 
            "swagger", "openapi"
        ]
        
        # Desktop app indicators
        desktop_indicators = [
            "electron", "qt", "gtk", "wxwidgets", "window", "mainwindow", "form"
        ]
        
        # Count indicators
        web_score = sum(1 for f in files if any(ind in f.lower() for ind in web_indicators))
        api_score = sum(1 for f in files if any(ind in f.lower() for ind in api_indicators))
        desktop_score = sum(1 for f in files if any(ind in f.lower() for ind in desktop_indicators))
        
        # Additional checks for specific files and content
        for f_path, code_file in files.items():
            # Check file content for indicators
            content = code_file.content.lower()
            
            if "<!doctype html>" in content or "<html" in content:
                web_score += 1
            
            if "api" in content and ("endpoint" in content or "route" in content):
                api_score += 1
            
            if "window" in content and ("gui" in content or "interface" in content):
                desktop_score += 1
        
        # Determine type based on scores
        max_score = max(web_score, api_score, desktop_score)
        
        if max_score == 0:
            # Check if it's a library/framework
            if any("setup.py" in f or "package.json" in f for f in files):
                return SystemType.LIBRARY
            return SystemType.UNKNOWN
        
        if max_score == web_score:
            return SystemType.WEB_APP
        elif max_score == api_score:
            return SystemType.API
        elif max_score == desktop_score:
            return SystemType.DESKTOP_APP
        
        return SystemType.UNKNOWN
    
    def _build_dependency_graph(self, system_info: SystemInfo) -> nx.DiGraph:
        """Build a dependency graph of files"""
        G = nx.DiGraph()
        
        # Add all files as nodes
        for file_path in system_info.files:
            G.add_node(file_path)
        
        # Add edges based on imports
        for file_path, code_file in system_info.files.items():
            for imported in code_file.imports:
                # Try to find the corresponding file
                for other_path, other_file in system_info.files.items():
                    if self._file_provides_import(other_file, imported):
                        G.add_edge(file_path, other_path)
        
        return G
    
    def _file_provides_import(self, code_file: CodeFile, import_name: str) -> bool:
        """Check if a file provides the given import"""
        # Very simple check for now
        file_basename = os.path.splitext(os.path.basename(code_file.path))[0]
        return file_basename == import_name or import_name.endswith(f".{file_basename}")
    
    def _analyze_dependencies(self, system_info: SystemInfo) -> None:
        """Analyze external dependencies"""
        # Extract dependencies from common dependency files
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            
            if "requirements.txt" in file_path:
                self._extract_python_dependencies(system_info, full_path)
            elif "package.json" in file_path:
                self._extract_npm_dependencies(system_info, full_path)
            elif "composer.json" in file_path:
                self._extract_composer_dependencies(system_info, full_path)
            elif "gemfile" in file_path.lower():
                self._extract_ruby_dependencies(system_info, full_path)
            elif "build.gradle" in file_path or "pom.xml" in file_path:
                self._extract_java_dependencies(system_info, full_path)
    
    def _extract_python_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Python dependencies from requirements.txt"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse dependency
                    parts = re.split(r'[=<>]', line, 1)
                    name = parts[0].strip()
                    version = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Add to dependencies
                    system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_npm_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract NPM dependencies from package.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['dependencies', 'devDependencies']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_composer_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract PHP dependencies from composer.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['require', 'require-dev']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_ruby_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Ruby dependencies from Gemfile"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Match gem declarations
                    match = re.match(r'gem\s+[\'"]([^\'"]+)[\'"](?:,\s*[\'"]([^\'"]+)[\'"])?', line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) or ""
                        
                        # Add to dependencies
                        system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_java_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Java dependencies from build.gradle or pom.xml"""
        try:
            if file_path.endswith("build.gradle"):
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Extract dependencies from build.gradle using regex
                    for match in re.finditer(r'(?:compile|implementation|api)\s+[\'"]([^:]+):([^:]+):([^\'"]+)[\'"]', content):
                        group = match.group(1)
                        name = match.group(2)
                        version = match.group(3)
                        
                        full_name = f"{group}:{name}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
            elif file_path.endswith("pom.xml"):
                # Simple regex-based extraction for POMs
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract dependencies from pom.xml
                    dependencies = re.findall(r'<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>', content)
                    for group, artifact, version in dependencies:
                        full_name = f"{group}:{artifact}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _identify_api_endpoints(self, system_info: SystemInfo) -> None:
        """Identify API endpoints"""
        # Only process if it's an API or web app
        if system_info.system_type not in [SystemType.API, SystemType.WEB_APP]:
            return
        
        endpoints = []
        
        for file_path, code_file in system_info.files.items():
            # Check based on language
            if code_file.language == LanguageType.PYTHON:
                self._extract_python_endpoints(code_file, endpoints)
            elif code_file.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                self._extract_js_endpoints(code_file, endpoints)
            elif code_file.language == LanguageType.JAVA:
                self._extract_java_endpoints(code_file, endpoints)
            # Add more languages as needed
        
        system_info.api_endpoints = endpoints
    
    def _extract_python_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Python file"""
        content = code_file.content
        
        # Flask endpoints
        for match in re.finditer(r'@(?:app|blueprint)\.route\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # Django URLs
        for match in re.finditer(r'path\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # FastAPI endpoints
        for match in re.finditer(r'@(?:app|router)\.(?:get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_js_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from JavaScript/TypeScript file"""
        content = code_file.content
        
        # Express.js endpoints
        for method in ['get', 'post', 'put', 'delete', 'patch']:
            for match in re.finditer(rf'(?:app|router)\.{method}\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
                endpoints.append(match.group(1))
        
        # Generic route definitions
        for match in re.finditer(r'route\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_java_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Java file"""
        content = code_file.content
        
        # Spring endpoints
        for match in re.finditer(r'@RequestMapping\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # JAX-RS endpoints
        for match in re.finditer(r'@Path\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _check_vulnerabilities(self, system_info: SystemInfo) -> None:
        """Check for known vulnerabilities"""
        vulnerabilities = []
        
        # In a real implementation, this would use a security database or API
        # For now, we'll look for some common vulnerability patterns
        security_patterns = {
            # SQL Injection
            r'(?:SELECT|INSERT|UPDATE|DELETE).*\+\s*["\']': "Potential SQL Injection",
            # XSS
            r'(?:innerHTML|document\.write)\s*\(': "Potential XSS vulnerability",
            # Hardcoded credentials
            r'(?:password|secret|key|token)\s*=\s*["\'][^"\']+["\']': "Hardcoded credentials",
            # Command injection
            r'(?:exec|spawn|system)\s*\(': "Potential command injection",
            # Insecure file operations
            r'eval\s*\(': "Insecure eval() usage"
        }
        
        for file_path, code_file in system_info.files.items():
            content = code_file.content
            
            for pattern, issue in security_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    vulnerability = f"{issue} in {file_path}"
                    vulnerabilities.append(vulnerability)
                    code_file.vulnerabilities.append(issue)
        
        # Also check for outdated dependencies with known vulnerabilities
        # In a real implementation, this would check against a vulnerability database
        
        system_info.vulnerabilities = vulnerabilities
    
    def _identify_database_connections(self, system_info: SystemInfo) -> None:
        """Identify database connections"""
        db_info = {}
        
        # Database patterns to look for
        db_patterns = {
            "mysql": r'mysql|mysqli|pdo_mysql',
            "postgres": r'postgres|pg_connect|pdo_pgsql',
            "sqlite": r'sqlite|pdo_sqlite',
            "mongodb": r'mongodb|mongo_connect',
            "oracle": r'oracle|oci_connect',
            "sqlserver": r'sqlserver|mssql|pdo_sqlsrv'
        }
        
        # Check configuration files first
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            try:
                with open(full_path, 'r') as f:
                    content = f.read().lower()
                    
                    # Look for connection strings
                    for db_type, pattern in db_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            if db_type not in db_info:
                                db_info[db_type] = []
                            db_info[db_type].append(file_path)
                            
                            # Look for connection parameters
                            for param in ["host", "port", "database", "dbname", "user", "username", "pass", "password"]:
                                matches = re.finditer(rf'{param}\s*[=:]\s*[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
                                for match in matches:
                                    if "connection_params" not in db_info:
                                        db_info["connection_params"] = {}
                                    db_info["connection_params"][param] = match.group(1)
            except Exception as e:
                logger.warning(f"Error checking database info in {file_path}: {str(e)}")
        
        # Also check code files
        for file_path, code_file in system_info.files.items():
            content = code_file.content.lower()
            
            # Look for database imports and connection code
            for db_type, pattern in db_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    if db_type not in db_info:
                        db_info[db_type] = []
                    db_info[db_type].append(file_path)
        
        system_info.database_info = db_info

class CodeTransformer(ABC):
    """Base class for code transformers"""
    
    @abstractmethod
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        pass
    
    @abstractmethod
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """
        Transform the code
        
        Args:
            code_file: Code file to transform
            system_info: System information
            
        Returns:
            Tuple of (transformed code, list of applied transformations)
        """
        pass

class PythonModernizer(CodeTransformer):
    """Modernizes Python code"""
    
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        return code_file.language == LanguageType.PYTHON
    
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """Transform Python code to modern standards"""
        content = code_file.content
        transformations = []
        
        # Add type hints
        content, type_transforms = self._add_type_hints(content)
        if type_transforms:
            transformations.append("Added type hints")
        
        # Convert to f-strings
        content, fstring_count = self._convert_to_fstrings(content)
        if fstring_count > 0:
            transformations.append(f"Converted {fstring_count} string formats to f-strings")
        
        # Use modern Python features
        content, modern_transforms = self._modernize_python_features(content)
        transformations.extend(modern_transforms)
        
        # Update imports
        content, import_transforms = self._update_imports(content, system_info)
        transformations.extend(import_transforms)
        
        return content, transformations
    
    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add type hints to Python code"""
        # This would require more sophisticated parsing
        # For a simple example, we'll just add typing import
        if 'from typing import ' not in content and 'import typing' not in content:
            content = "from typing import List, Dict, Tuple, Optional, Any, Union\n" + content
            return content, ["Added typing imports"]
        return content, []
    
    def _convert_to_fstrings(self, content: str) -> Tuple[str, int]:
        """Convert old-style string formatting to f-strings"""
        # Convert .format() style
        pattern = r'([\'"].*?[\'"])\s*\.\s*format\s*\((.*?)\)'
        
        count = 0
        for match in re.finditer(pattern, content):
            old_str = match.group(0)
            string_content = match.group(1)[1:-1]  # Remove quotes
            format_args = match.group(2)
            
            # Simple conversion for basic cases
            if not format_args.strip():
                continue
                
            # Try to convert
            try:
                # If format args are simple like "var1, var2"
                if re.match(r'^[\w\s,]+#!/usr/bin/env python3
"""
Kaleidoscope AI - System Upgrade Module
=======================================
Automated system for upgrading and modernizing outdated codebases.
Preserves functionality while enhancing architecture, security, and performance.
"""

import os
import sys
import re
import ast
import json
import shutil
import tempfile
import subprocess
import importlib
import logging
import zipfile
import tarfile
import uuid
import hashlib
import datetime
import docker
import multiprocessing
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_upgrade.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    CSHARP = auto()
    CPP = auto()
    RUBY = auto()
    PHP = auto()
    GO = auto()
    RUST = auto()
    SWIFT = auto()
    KOTLIN = auto()
    UNKNOWN = auto()

class SystemType(Enum):
    """Types of systems to upgrade"""
    WEB_APP = auto()
    DESKTOP_APP = auto()
    MOBILE_APP = auto()
    API = auto()
    CLI = auto()
    LIBRARY = auto()
    FRAMEWORK = auto()
    DATABASE = auto()
    UNKNOWN = auto()

class UpgradeStrategy(Enum):
    """Strategies for system upgrades"""
    IN_PLACE = auto()  # Modify existing codebase
    INCREMENTAL = auto()  # Upgrade component by component
    FULL_REWRITE = auto()  # Complete rewrite with same language
    LANGUAGE_MIGRATION = auto()  # Rewrite in different language
    WRAPPER = auto()  # Create wrapper around existing system

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version: str
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    is_vulnerable: bool = False
    vulnerability_details: Optional[str] = None
    is_outdated: bool = False
    upgrade_path: Optional[str] = None
    is_deprecated: bool = False
    alternatives: List[str] = field(default_factory.list)

@dataclass
class CodeFile:
    """Information about a code file"""
    path: str
    language: LanguageType
    content: str
    ast: Optional[Any] = None
    imports: List[str] = field(default_factory.list)
    exports: List[str] = field(default_factory.list)
    dependencies: List[DependencyInfo] = field(default_factory.list)
    vulnerabilities: List[str] = field(default_factory.list)
    outdated_patterns: List[str] = field.default_factory.list)
    complexity_score: float = 0.0
    is_test: bool = False

@dataclass
class SystemInfo:
    """Information about the system to upgrade"""
    root_path: str
    system_type: SystemType
    primary_language: LanguageType
    other_languages: List[LanguageType] = field(default_factory.list)
    files: Dict[str, CodeFile] = field(default_factory.dict)
    dependencies: Dict[str, DependencyInfo] = field(default_factory.dict)
    entry_points: List[str] = field(default_factory.list)
    config_files: List[str] = field(default_factory.list)
    database_info: Dict[str, Any] = field(default_factory.dict)
    api_endpoints: List[str] = field(default_factory.list)
    vulnerabilities: List[str] = field.default_factory.list)
    dependencies_graph: Optional[nx.DiGraph] = None
    file_count: int = 0
    code_size: int = 0  # In bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "root_path": self.root_path,
            "system_type": self.system_type.name,
            "primary_language": self.primary_language.name,
            "other_languages": [lang.name for lang in self.other_languages],
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "database_info": self.database_info,
            "api_endpoints": self.api_endpoints,
            "vulnerabilities": self.vulnerabilities,
            "file_count": self.file_count,
            "code_size": self.code_size,
            "dependencies": {k: {"name": v.name, "version": v.version} for k, v in self.dependencies.items()},
        }
        return result

@dataclass
class UpgradeConfig:
    """Configuration for the upgrade process"""
    target_language: LanguageType
    strategy: UpgradeStrategy
    preserve_functionality: bool = True
    update_dependencies: bool = True
    fix_vulnerabilities: bool = True
    improve_performance: bool = True
    add_tests: bool = True
    modernize_architecture: bool = True
    refactor_code: bool = True
    target_frameworks: List[str] = field(default_factory.list)
    excluded_paths: List[str] = field.default_factory.list)
    keep_original: bool = True
    max_parallel_processes: int = multiprocessing.cpu_count()
    timeout_seconds: int = 3600  # 1 hour

@dataclass
class UpgradeResult:
    """Results of the upgrade process"""
    success: bool
    output_path: str
    strategy_used: UpgradeStrategy
    upgraded_files: List[str] = field.default_factory.list)
    errors: List[str] = field.default_factory.list)
    backup_path: Optional[str] = None
    time_taken_seconds: float = 0.0
    size_difference: int = 0  # Difference in bytes
    applied_transformations: List[str] = field.default_factory.list)
    license_path: Optional[str] = None

class LanguageDetector:
    """Detects programming languages from file content and extensions"""
    
    def __init__(self):
        """Initialize language detector"""
        self.extension_map = {
            ".py": LanguageType.PYTHON,
            ".js": LanguageType.JAVASCRIPT,
            ".jsx": LanguageType.JAVASCRIPT,
            ".ts": LanguageType.TYPESCRIPT,
            ".tsx": LanguageType.TYPESCRIPT,
            ".java": LanguageType.JAVA,
            ".cs": LanguageType.CSHARP,
            ".cpp": LanguageType.CPP,
            ".cc": LanguageType.CPP,
            ".cxx": LanguageType.CPP,
            ".c": LanguageType.CPP,
            ".h": LanguageType.CPP,
            ".hpp": LanguageType.CPP,
            ".rb": LanguageType.RUBY,
            ".php": LanguageType.PHP,
            ".go": LanguageType.GO,
            ".rs": LanguageType.RUST,
            ".swift": LanguageType.SWIFT,
            ".kt": LanguageType.KOTLIN
        }
        
        self.shebang_patterns = {
            r"^\s*#!.*python": LanguageType.PYTHON,
            r"^\s*#!.*node": LanguageType.JAVASCRIPT,
            r"^\s*#!.*ruby": LanguageType.RUBY,
            r"^\s*#!.*php": LanguageType.PHP
        }
        
        self.content_patterns = {
            r"import\s+[a-zA-Z0-9_]+|from\s+[a-zA-Z0-9_\.]+\s+import": LanguageType.PYTHON,
            r"require\s*\(\s*['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*\)|import\s+[a-zA-Z0-9_]+\s+from": LanguageType.JAVASCRIPT,
            r"import\s+{\s*[a-zA-Z0-9_,\s]+\s*}\s+from|interface\s+[a-zA-Z0-9_]+": LanguageType.TYPESCRIPT,
            r"public\s+class|import\s+java\.": LanguageType.JAVA,
            r"namespace\s+[a-zA-Z0-9_\.]+|using\s+[a-zA-Z0-9_\.]+;": LanguageType.CSHARP,
            r"#include\s*<[a-zA-Z0-9_\.]+>|#include\s*\"[a-zA-Z0-9_\.]+\"": LanguageType.CPP,
            r"require\s+['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*|def\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUBY,
            r"<\?php|namespace\s+[a-zA-Z0-9_\\]+;": LanguageType.PHP,
            r"package\s+[a-zA-Z0-9_]+|func\s+[a-zA-Z0-9_]+\s*\(": LanguageType.GO,
            r"use\s+[a-zA-Z0-9_:]+|fn\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUST,
            r"import\s+[a-zA-Z0-9_\.]+|class\s+[a-zA-Z0-9_]+\s*:": LanguageType.SWIFT,
            r"package\s+[a-zA-Z0-9_\.]+|fun\s+[a-zA-Z0-9_]+\s*\(": LanguageType.KOTLIN
        }
    
    def detect_language(self, file_path: str, content: Optional[str] = None) -> LanguageType:
        """
        Detect the programming language of a file
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Detected language type
        """
        # Try by extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # If no content provided, try to read it
        if content is None:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {str(e)}")
                return LanguageType.UNKNOWN
        
        # Try by shebang
        for pattern, lang in self.shebang_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        # Try by content patterns
        for pattern, lang in self.content_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        return LanguageType.UNKNOWN

class SystemAnalyzer:
    """Analyzes a system to gather information needed for upgrading"""
    
    def __init__(self):
        """Initialize system analyzer"""
        self.language_detector = LanguageDetector()
        self.excluded_dirs = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__", 
            "venv", "env", ".env", ".venv", "dist", "build"
        }
        self.excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"
        }
    
    def analyze_system(self, path: str) -> SystemInfo:
        """
        Analyze a system to gather information
        
        Args:
            path: Path to the system root directory
            
        Returns:
            System information
        """
        logger.info(f"Analyzing system at {path}")
        
        # Initialize system info
        system_info = SystemInfo(
            root_path=path,
            system_type=SystemType.UNKNOWN,
            primary_language=LanguageType.UNKNOWN
        )
        
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        
        # Count languages for later determining primary language
        language_counts = {}
        
        # Walk through the directory tree
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path, topdown=True):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Process each file
            for file in files:
                if file in self.excluded_files:
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, path)
                
                # Skip binary files and large files
                if self._is_binary_file(file_path) or os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                    continue
                
                try:
                    # Read file content
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                    
                    # Detect language
                    language = self.language_detector.detect_language(file_path, content)
                    
                    # Update language counts
                    if language != LanguageType.UNKNOWN:
                        language_counts[language] = language_counts.get(language, 0) + 1
                    
                    # Create code file info
                    code_file = CodeFile(
                        path=relative_path,
                        language=language,
                        content=content
                    )
                    
                    # Extract imports and other information based on language
                    self._extract_file_info(code_file)
                    
                    # Add to system info
                    system_info.files[relative_path] = code_file
                    
                    # Update total size
                    file_size = len(content.encode('utf-8'))
                    total_size += file_size
                    file_count += 1
                    
                    # Check for special files
                    file_lower = file.lower()
                    if any(name in file_lower for name in ["readme", "license", "dockerfile", "docker-compose"]):
                        # Could add special handling here
                        pass
                    
                    # Identify potential entry points
                    if self._is_entry_point(file_path, relative_path, language):
                        system_info.entry_points.append(relative_path)
                    
                    # Identify configuration files
                    if self._is_config_file(file_path, relative_path):
                        system_info.config_files.append(relative_path)
                
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
        
        # Set primary language and other languages
        if language_counts:
            primary_language = max(language_counts.items(), key=lambda x: x[1])[0]
            system_info.primary_language = primary_language
            system_info.other_languages = [lang for lang in language_counts.keys() if lang != primary_language]
        
        # Determine system type
        system_info.system_type = self._determine_system_type(system_info)
        
        # Update file count and code size
        system_info.file_count = file_count
        system_info.code_size = total_size
        
        # Build dependency graph
        system_info.dependencies_graph = self._build_dependency_graph(system_info)
        
        # Analyze dependencies
        self._analyze_dependencies(system_info)
        
        # Identify API endpoints
        self._identify_api_endpoints(system_info)
        
        # Check for vulnerabilities
        self._check_vulnerabilities(system_info)
        
        # Identify database connections
        self._identify_database_connections(system_info)
        
        logger.info(f"System analysis complete: {system_info.primary_language.name}, {system_info.system_type.name}, {file_count} files")
        
        return system_info
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary"""
        try:
            with open(file_path, 'r') as f:
                f.read(1024)
            return False
        except UnicodeDecodeError:
            return True
    
    def _extract_file_info(self, code_file: CodeFile) -> None:
        """Extract imports and other information from file"""
        language = code_file.language
        content = code_file.content
        
        if language == LanguageType.PYTHON:
            self._extract_python_imports(code_file)
        elif language == LanguageType.JAVASCRIPT:
            self._extract_javascript_imports(code_file)
        elif language == LanguageType.TYPESCRIPT:
            self._extract_typescript_imports(code_file)
        elif language == LanguageType.JAVA:
            self._extract_java_imports(code_file)
    
    def _extract_python_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Python file"""
        try:
            tree = ast.parse(code_file.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        code_file.imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        for name in node.names:
                            code_file.imports.append(f"{module_name}.{name.name}")
        except SyntaxError:
            # Fall back to regex for invalid Python
            for match in re.finditer(r'^\s*(?:import|from)\s+([\w\.]+)', code_file.content, re.MULTILINE):
                code_file.imports.append(match.group(1))
    
    def _extract_javascript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from JavaScript file"""
        # ES6 imports
        for match in re.finditer(r'import\s+(?:{\s*([\w\s,]+)\s*}|(\w+))\s+from\s+[\'"]([^\'"]*)[\'"]\s*;?', code_file.content):
            if match.group(1):  # Named imports
                for name in match.group(1).split(','):
                    code_file.imports.append(name.strip())
            elif match.group(2):  # Default import
                code_file.imports.append(match.group(2))
        
        # CommonJS requires
        for match in re.finditer(r'(?:const|let|var)\s+([\w{}:\s,]+)\s*=\s*require\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)\s*;?', code_file.content):
            code_file.imports.append(match.group(2))
    
    def _extract_typescript_imports(self, code_file: CodeFile) -> None:
        """Extract imports from TypeScript file"""
        # TypeScript imports are similar to JavaScript
        self._extract_javascript_imports(code_file)
    
    def _extract_java_imports(self, code_file: CodeFile) -> None:
        """Extract imports from Java file"""
        for match in re.finditer(r'import\s+([\w\.]+)(?:\.\*)?;', code_file.content):
            code_file.imports.append(match.group(1))
    
    def _is_entry_point(self, file_path: str, relative_path: str, language: LanguageType) -> bool:
        """Identify if a file is an entry point"""
        file_name = os.path.basename(file_path).lower()
        
        # Common entry point patterns
        if language == LanguageType.PYTHON:
            return file_name in ["main.py", "app.py", "manage.py", "run.py"] or "if __name__ == '__main__'" in open(file_path, 'r', errors='ignore').read()
        elif language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
            return file_name in ["index.js", "main.js", "app.js", "server.js", "index.ts", "main.ts", "app.ts", "server.ts"]
        elif language == LanguageType.JAVA:
            return "public static void main(" in open(file_path, 'r', errors='ignore').read()
        elif language == LanguageType.CSHARP:
            return "static void Main(" in open(file_path, 'r', errors='ignore').read() or "Program.cs" in file_path
        
        return False
    
    def _is_config_file(self, file_path: str, relative_path: str) -> bool:
        """Identify if a file is a configuration file"""
        file_name = os.path.basename(file_path).lower()
        ext = os.path.splitext(file_name)[1].lower()
        
        config_patterns = [
            "config", "settings", ".env", ".ini", ".yml", ".yaml", ".json", ".xml", ".toml",
            "package.json", "composer.json", "pyproject.toml", "requirements.txt", "Gemfile",
            ".gitignore", "Dockerfile", "docker-compose"
        ]
        
        return any(pattern in file_name for pattern in config_patterns)
    
    def _determine_system_type(self, system_info: SystemInfo) -> SystemType:
        """Determine the type of system"""
        files = system_info.files
        
        # Web app indicators
        web_indicators = [
            "index.html", "app.js", "webpack.config.js", "package.json", 
            "views", "templates", "public", "static", "assets"
        ]
        
        # API indicators
        api_indicators = [
            "routes", "controllers", "endpoints", "api", "rest", "graphql", 
            "swagger", "openapi"
        ]
        
        # Desktop app indicators
        desktop_indicators = [
            "electron", "qt", "gtk", "wxwidgets", "window", "mainwindow", "form"
        ]
        
        # Count indicators
        web_score = sum(1 for f in files if any(ind in f.lower() for ind in web_indicators))
        api_score = sum(1 for f in files if any(ind in f.lower() for ind in api_indicators))
        desktop_score = sum(1 for f in files if any(ind in f.lower() for ind in desktop_indicators))
        
        # Additional checks for specific files and content
        for f_path, code_file in files.items():
            # Check file content for indicators
            content = code_file.content.lower()
            
            if "<!doctype html>" in content or "<html" in content:
                web_score += 1
            
            if "api" in content and ("endpoint" in content or "route" in content):
                api_score += 1
            
            if "window" in content and ("gui" in content or "interface" in content):
                desktop_score += 1
        
        # Determine type based on scores
        max_score = max(web_score, api_score, desktop_score)
        
        if max_score == 0:
            # Check if it's a library/framework
            if any("setup.py" in f or "package.json" in f for f in files):
                return SystemType.LIBRARY
            return SystemType.UNKNOWN
        
        if max_score == web_score:
            return SystemType.WEB_APP
        elif max_score == api_score:
            return SystemType.API
        elif max_score == desktop_score:
            return SystemType.DESKTOP_APP
        
        return SystemType.UNKNOWN
    
    def _build_dependency_graph(self, system_info: SystemInfo) -> nx.DiGraph:
        """Build a dependency graph of files"""
        G = nx.DiGraph()
        
        # Add all files as nodes
        for file_path in system_info.files:
            G.add_node(file_path)
        
        # Add edges based on imports
        for file_path, code_file in system_info.files.items():
            for imported in code_file.imports:
                # Try to find the corresponding file
                for other_path, other_file in system_info.files.items():
                    if self._file_provides_import(other_file, imported):
                        G.add_edge(file_path, other_path)
        
        return G
    
    def _file_provides_import(self, code_file: CodeFile, import_name: str) -> bool:
        """Check if a file provides the given import"""
        # Very simple check for now
        file_basename = os.path.splitext(os.path.basename(code_file.path))[0]
        return file_basename == import_name or import_name.endswith(f".{file_basename}")
    
    def _analyze_dependencies(self, system_info: SystemInfo) -> None:
        """Analyze external dependencies"""
        # Extract dependencies from common dependency files
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            
            if "requirements.txt" in file_path:
                self._extract_python_dependencies(system_info, full_path)
            elif "package.json" in file_path:
                self._extract_npm_dependencies(system_info, full_path)
            elif "composer.json" in file_path:
                self._extract_composer_dependencies(system_info, full_path)
            elif "gemfile" in file_path.lower():
                self._extract_ruby_dependencies(system_info, full_path)
            elif "build.gradle" in file_path or "pom.xml" in file_path:
                self._extract_java_dependencies(system_info, full_path)
    
    def _extract_python_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Python dependencies from requirements.txt"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse dependency
                    parts = re.split(r'[=<>]', line, 1)
                    name = parts[0].strip()
                    version = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Add to dependencies
                    system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_npm_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract NPM dependencies from package.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['dependencies', 'devDependencies']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_composer_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract PHP dependencies from composer.json"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process dependencies
                for dep_type in ['require', 'require-dev']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            # Add to dependencies
                            system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_ruby_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Ruby dependencies from Gemfile"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Match gem declarations
                    match = re.match(r'gem\s+[\'"]([^\'"]+)[\'"](?:,\s*[\'"]([^\'"]+)[\'"])?', line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) or ""
                        
                        # Add to dependencies
                        system_info.dependencies[name] = DependencyInfo(name=name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _extract_java_dependencies(self, system_info: SystemInfo, file_path: str) -> None:
        """Extract Java dependencies from build.gradle or pom.xml"""
        try:
            if file_path.endswith("build.gradle"):
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Extract dependencies from build.gradle using regex
                    for match in re.finditer(r'(?:compile|implementation|api)\s+[\'"]([^:]+):([^:]+):([^\'"]+)[\'"]', content):
                        group = match.group(1)
                        name = match.group(2)
                        version = match.group(3)
                        
                        full_name = f"{group}:{name}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
            elif file_path.endswith("pom.xml"):
                # Simple regex-based extraction for POMs
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract dependencies from pom.xml
                    dependencies = re.findall(r'<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>', content)
                    for group, artifact, version in dependencies:
                        full_name = f"{group}:{artifact}"
                        system_info.dependencies[full_name] = DependencyInfo(name=full_name, version=version)
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {str(e)}")
    
    def _identify_api_endpoints(self, system_info: SystemInfo) -> None:
        """Identify API endpoints"""
        # Only process if it's an API or web app
        if system_info.system_type not in [SystemType.API, SystemType.WEB_APP]:
            return
        
        endpoints = []
        
        for file_path, code_file in system_info.files.items():
            # Check based on language
            if code_file.language == LanguageType.PYTHON:
                self._extract_python_endpoints(code_file, endpoints)
            elif code_file.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                self._extract_js_endpoints(code_file, endpoints)
            elif code_file.language == LanguageType.JAVA:
                self._extract_java_endpoints(code_file, endpoints)
            # Add more languages as needed
        
        system_info.api_endpoints = endpoints
    
    def _extract_python_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Python file"""
        content = code_file.content
        
        # Flask endpoints
        for match in re.finditer(r'@(?:app|blueprint)\.route\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # Django URLs
        for match in re.finditer(r'path\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # FastAPI endpoints
        for match in re.finditer(r'@(?:app|router)\.(?:get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_js_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from JavaScript/TypeScript file"""
        content = code_file.content
        
        # Express.js endpoints
        for method in ['get', 'post', 'put', 'delete', 'patch']:
            for match in re.finditer(rf'(?:app|router)\.{method}\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
                endpoints.append(match.group(1))
        
        # Generic route definitions
        for match in re.finditer(r'route\s*\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _extract_java_endpoints(self, code_file: CodeFile, endpoints: List[str]) -> None:
        """Extract API endpoints from Java file"""
        content = code_file.content
        
        # Spring endpoints
        for match in re.finditer(r'@RequestMapping\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
        
        # JAX-RS endpoints
        for match in re.finditer(r'@Path\(\s*[\'"]([^\'"]+)[\'"]', content):
            endpoints.append(match.group(1))
    
    def _check_vulnerabilities(self, system_info: SystemInfo) -> None:
        """Check for known vulnerabilities"""
        vulnerabilities = []
        
        # In a real implementation, this would use a security database or API
        # For now, we'll look for some common vulnerability patterns
        security_patterns = {
            # SQL Injection
            r'(?:SELECT|INSERT|UPDATE|DELETE).*\+\s*["\']': "Potential SQL Injection",
            # XSS
            r'(?:innerHTML|document\.write)\s*\(': "Potential XSS vulnerability",
            # Hardcoded credentials
            r'(?:password|secret|key|token)\s*=\s*["\'][^"\']+["\']': "Hardcoded credentials",
            # Command injection
            r'(?:exec|spawn|system)\s*\(': "Potential command injection",
            # Insecure file operations
            r'eval\s*\(': "Insecure eval() usage"
        }
        
        for file_path, code_file in system_info.files.items():
            content = code_file.content
            
            for pattern, issue in security_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    vulnerability = f"{issue} in {file_path}"
                    vulnerabilities.append(vulnerability)
                    code_file.vulnerabilities.append(issue)
        
        # Also check for outdated dependencies with known vulnerabilities
        # In a real implementation, this would check against a vulnerability database
        
        system_info.vulnerabilities = vulnerabilities
    
    def _identify_database_connections(self, system_info: SystemInfo) -> None:
        """Identify database connections"""
        db_info = {}
        
        # Database patterns to look for
        db_patterns = {
            "mysql": r'mysql|mysqli|pdo_mysql',
            "postgres": r'postgres|pg_connect|pdo_pgsql',
            "sqlite": r'sqlite|pdo_sqlite',
            "mongodb": r'mongodb|mongo_connect',
            "oracle": r'oracle|oci_connect',
            "sqlserver": r'sqlserver|mssql|pdo_sqlsrv'
        }
        
        # Check configuration files first
        for file_path in system_info.config_files:
            full_path = os.path.join(system_info.root_path, file_path)
            try:
                with open(full_path, 'r') as f:
                    content = f.read().lower()
                    
                    # Look for connection strings
                    for db_type, pattern in db_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            if db_type not in db_info:
                                db_info[db_type] = []
                            db_info[db_type].append(file_path)
                            
                            # Look for connection parameters
                            for param in ["host", "port", "database", "dbname", "user", "username", "pass", "password"]:
                                matches = re.finditer(rf'{param}\s*[=:]\s*[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
                                for match in matches:
                                    if "connection_params" not in db_info:
                                        db_info["connection_params"] = {}
                                    db_info["connection_params"][param] = match.group(1)
            except Exception as e:
                logger.warning(f"Error checking database info in {file_path}: {str(e)}")
        
        # Also check code files
        for file_path, code_file in system_info.files.items():
            content = code_file.content.lower()
            
            # Look for database imports and connection code
            for db_type, pattern in db_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    if db_type not in db_info:
                        db_info[db_type] = []
                    db_info[db_type].append(file_path)
        
        system_info.database_info = db_info

class CodeTransformer(ABC):
    """Base class for code transformers"""
    
    @abstractmethod
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        pass
    
    @abstractmethod
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """
        Transform the code
        
        Args:
            code_file: Code file to transform
            system_info: System information
            
        Returns:
            Tuple of (transformed code, list of applied transformations)
        """
        pass

class PythonModernizer(CodeTransformer):
    """Modernizes Python code"""
    
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        return code_file.language == LanguageType.PYTHON
    
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """Transform Python code to modern standards"""
        content = code_file.content
        transformations = []
        
        # Add type hints
        content, type_transforms = self._add_type_hints(content)
        if type_transforms:
            transformations.append("Added type hints")
        
        # Convert to f-strings
        content, fstring_count = self._convert_to_fstrings(content)
        if fstring_count > 0:
            transformations.append(f"Converted {fstring_count} string formats to f-strings")
        
        # Use modern Python features
        content, modern_transforms = self._modernize_python_features(content)
        transformations.extend(modern_transforms)
        
        # Update imports
        content, import_transforms = self._update_imports(content, system_info)
        transformations.extend(import_transforms)
        
        return content, transformations
    
    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add type hints to Python code"""
        # This would require more sophisticated parsing
        # For a simple example, we'll just add typing import
        if 'from typing import ' not in content and 'import typing' not in content:
            content = "from typing import List, Dict, Tuple, Optional, Any, Union\n" + content
            return content, ["Added typing imports"]
        return content, []
    
    def _convert_to_fstrings(self, content: str) -> Tuple[str, int]:
        """Convert old-style string formatting to f-strings"""
        # Convert .format() style
        pattern = r'([\'"].*?[\'"])\s*\.\s*format\s*\((.*?)\)'
        
        count = 0
        for match in re.finditer(pattern, content):
            old_str = match.group(0)
            string_content = match.group(1)[1:-1]  # Remove quotes
            format_args = match.group(2)
            
            # Simple conversion for basic cases
            if not format_args.strip():
                continue
                
            # Try to convert
            try:
                # If format args are simple like "var1, var2"
                if re.match(r'^[\w\s,]+#!/usr/bin/env python3
"""
Kaleidoscope AI - System Upgrade Module
=======================================
Automated system for upgrading and modernizing outdated codebases.
Preserves functionality while enhancing architecture, security, and performance.
"""

import os
import sys
import re
import ast
import json
import shutil
import tempfile
import subprocess
import importlib
import logging
import zipfile
import tarfile
import uuid
import hashlib
import datetime
import docker
import multiprocessing
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_upgrade.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    CSHARP = auto()
    CPP = auto()
    RUBY = auto()
    PHP = auto()
    GO = auto()
    RUST = auto()
    SWIFT = auto()
    KOTLIN = auto()
    UNKNOWN = auto()

class SystemType(Enum):
    """Types of systems to upgrade"""
    WEB_APP = auto()
    DESKTOP_APP = auto()
    MOBILE_APP = auto()
    API = auto()
    CLI = auto()
    LIBRARY = auto()
    FRAMEWORK = auto()
    DATABASE = auto()
    UNKNOWN = auto()

class UpgradeStrategy(Enum):
    """Strategies for system upgrades"""
    IN_PLACE = auto()  # Modify existing codebase
    INCREMENTAL = auto()  # Upgrade component by component
    FULL_REWRITE = auto()  # Complete rewrite with same language
    LANGUAGE_MIGRATION = auto()  # Rewrite in different language
    WRAPPER = auto()  # Create wrapper around existing system

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version: str
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    is_vulnerable: bool = False
    vulnerability_details: Optional[str] = None
    is_outdated: bool = False
    upgrade_path: Optional[str] = None
    is_deprecated: bool = False
    alternatives: List[str] = field(default_factory.list)

@dataclass
class CodeFile:
    """Information about a code file"""
    path: str
    language: LanguageType
    content: str
    ast: Optional[Any] = None
    imports: List[str] = field(default_factory.list)
    exports: List[str] = field(default_factory.list)
    dependencies: List[DependencyInfo] = field.default_factory.list)
    vulnerabilities: List[str] = field.default_factory.list)
    outdated_patterns: List[str] = field.default_factory.list)
    complexity_score: float = 0.0
    is_test: bool = False

@dataclass
class SystemInfo:
    """Information about the system to upgrade"""
    root_path: str
    system_type: SystemType
    primary_language: LanguageType
    other_languages: List[LanguageType] = field.default_factory.list)
    files: Dict[str, CodeFile] = field.default_factory.dict)
    dependencies: Dict[str, DependencyInfo] = field.default_factory.dict)
    entry_points: List[str] = field.default_factory.list)
    config_files: List[str] = field.default_factory.list)
    database_info: Dict[str, Any] = field.default_factory.dict)
    api_endpoints: List[str] = field.default_factory.list)
    vulnerabilities: List[str] = field.default_factory.list)
    dependencies_graph: Optional[nx.DiGraph] = None
    file_count: int = 0
    code_size: int = 0  # In bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "root_path": self.root_path,
            "system_type": self.system_type.name,
            "primary_language": self.primary_language.name,
            "other_languages": [lang.name for lang in self.other_languages],
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "database_info": self.database_info,
            "api_endpoints": self.api_endpoints,
            "vulnerabilities": self.vulnerabilities,
            "file_count": self.file_count,
            "code_size": self.code_size,
            "dependencies": {k: {"name": v.name, "version": v.version} for k, v in self.dependencies.items()},
        }
        return result

@dataclass
class UpgradeConfig:
    """Configuration for the upgrade process"""
    target_language: LanguageType
    strategy: UpgradeStrategy
    preserve_functionality: bool = True
    update_dependencies: bool = True
    fix_vulnerabilities: bool = True
    improve_performance: bool = True
    add_tests: bool = True
    modernize_architecture: bool = True
    refactor_code: bool = True
    target_frameworks: List[str] = field.default_factory.list)
    excluded_paths: List[str] = field.default_factory.list)
    keep_original: bool = True
    max_parallel_processes: int = multiprocessing.cpu_count()
    timeout_seconds: int = 3600  # 1 hour

@dataclass
class UpgradeResult:
    """Results of the upgrade process"""
    success: bool
    output_path: str
    strategy_used: UpgradeStrategy
    upgraded_files: List[str] = field.default_factory.list)
    errors: List[str] = field.default_factory.list)
    backup_path: Optional[str] = None
    time_taken_seconds: float = 0.0
    size_difference: int = 0  # Difference in bytes
    applied_transformations: List[str] = field.default_factory.list)
    license_path: Optional[str] = None

class LanguageDetector:
    """Detects programming languages from file content and extensions"""
    
    def __init__(self):
        """Initialize language detector"""
        self.extension_map = {
            ".py": LanguageType.PYTHON,
            ".js": LanguageType.JAVASCRIPT,
            ".jsx": LanguageType.JAVASCRIPT,
            ".ts": LanguageType.TYPESCRIPT,
            ".tsx": LanguageType.TYPESCRIPT,
            ".java": LanguageType.JAVA,
            ".cs": LanguageType.CSHARP,
            ".cpp": LanguageType.CPP,
            ".cc": LanguageType.CPP,
            ".cxx": LanguageType.CPP,
            ".c": LanguageType.CPP,
            ".h": LanguageType.CPP,
            ".hpp": LanguageType.CPP,
            ".rb": LanguageType.RUBY,
            ".php": LanguageType.PHP,
            ".go": LanguageType.GO,
            ".rs": LanguageType.RUST,
            ".swift": LanguageType.SWIFT,
            ".kt": LanguageType.KOTLIN
        }
        
        self.shebang_patterns = {
            r"^\s*#!.*python": LanguageType.PYTHON,
            r"^\s*#!.*node": LanguageType.JAVASCRIPT,
            r"^\s*#!.*ruby": LanguageType.RUBY,
            r"^\s*#!.*php": LanguageType.PHP
        }
        
        self.content_patterns = {
            r"import\s+[a-zA-Z0-9_]+|from\s+[a-zA-Z0-9_\.]+\s+import": LanguageType.PYTHON,
            r"require\s*\(\s*['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*\)|import\s+[a-zA-Z0-9_]+\s+from": LanguageType.JAVASCRIPT,
            r"import\s+{\s*[a-zA-Z0-9_,\s]+\s*}\s+from|interface\s+[a-zA-Z0-9_]+": LanguageType.TYPESCRIPT,
            r"public\s+class|import\s+java\.": LanguageType.JAVA,
            r"namespace\s+[a-zA-Z0-9_\.]+|using\s+[a-zA-Z0-9_\.]+;": LanguageType.CSHARP,
            r"#include\s*<[a-zA-Z0-9_\.]+>|#include\s*\"[a-zA-Z0-9_\.]+\"": LanguageType.CPP,
            r"require\s+['\"][a-zA-Z0-9_\-\.\/]+['\"]\s*|def\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUBY,
            r"<\?php|namespace\s+[a-zA-Z0-9_\\]+;": LanguageType.PHP,
            r"package\s+[a-zA-Z0-9_]+|func\s+[a-zA-Z0-9_]+\s*\(": LanguageType.GO,
            r"use\s+[a-zA-Z0-9_:]+|fn\s+[a-zA-Z0-9_]+\s*\(": LanguageType.RUST,
            r"import\s+[a-zA-Z0-9_\.]+|class\s+[a-zA-Z0-9_]+\s*:": LanguageType.SWIFT,
            r"package\s+[a-zA-Z0-9_\.]+|fun\s+[a-zA-Z0-9_]+\s*\(": LanguageType.KOTLIN
        }
    
    def detect_language(self, file_path: str, content: Optional[str] = None) -> LanguageType:
        """
        Detect the programming language of a file
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Detected language type
        """
        # Try by extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # If no content provided, try to read it
        if content is None:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {str(e)}")
                return LanguageType.UNKNOWN
        
        # Try by shebang
        for pattern, lang in self.shebang_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        # Try by content patterns
        for pattern, lang in self.content_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                return lang
        
        return LanguageType.UNKNOWN

class SystemAnalyzer:
    """Analyzes a system to gather information needed for upgrading"""
    
    def __init__(self):
        """Initialize system analyzer"""
        self.language_detector = LanguageDetector()
        self.excluded_dirs = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__", 
            "venv", "env", ".env", ".venv", "dist", "build"
        }
        self.excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"
        }
    
    def analyze_system(self, path: str) -> SystemInfo:
        """
        Analyze a system to gather information
        
        Args:
            path: Path to the system root directory
            
        Returns:
            System information
        """
        logger.info(f"Analyzing system at {path}")
        
        # Initialize system info
        system_info = SystemInfo(
            root_path=path,
            system_type=SystemType.UNKNOWN,
            primary_language=LanguageType.UNKNOWN
        )
        
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        
        # Count languages for later determining primary language
        language_counts = {}
        
        # Walk through the directory tree
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path, topdown=True):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Process each file
            for file in files:
                if file in self.excluded_files:
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, path)
                