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
            logger.warning(f"Error parsing {file_path}: {str(e)}"), format_args):
                    vars = [v.strip() for v in format_args.split(',')]
                    placeholder_pattern = r'{\s*(\d+)\s*}'
                    
                    # Replace {0}, {1}, etc. with variable names
                    new_str = string_content
                    for i, var in enumerate(vars):
                        new_str = re.sub(placeholder_pattern.format(i), "{" + var + "}", new_str)
                    
                    # Create f-string
                    new_str = f'f"{new_str}"'
                    content = content.replace(old_str, new_str)
                    count += 1
            except:
                # Skip complex cases for this example
                pass
        
        # Convert % style
        pattern = r'([\'"].*?[\'"])\s*%\s*(.+)'
        for match in re.finditer(pattern, content):
            old_str = match.group(0)
            string_content = match.group(1)[1:-1]  # Remove quotes
            format_args = match.group(2)
            
            # Simple tuple case like "Hello %s, you are %d" % (name, age)
            if format_args.strip().startswith('(') and format_args.strip().endswith(')'):
                try:
                    # Extract variables from tuple
                    vars_str = format_args.strip()[1:-1]
                    vars = [v.strip() for v in vars_str.split(',')]
                    
                    # Replace %s, %d, etc. with {var}
                    new_str = string_content
                    placeholders = re.findall(r'%[sdifr]', string_content)
                    
                    for i, placeholder in enumerate(placeholders):
                        if i < len(vars):
                            new_str = new_str.replace(placeholder, "{" + vars[i] + "}", 1)
                    
                    # Create f-string
                    new_str = f'f"{new_str}"'
                    content = content.replace(old_str, new_str)
                    count += 1
                except:
                    # Skip complex cases
                    pass
        
        return content, count
    
    def _modernize_python_features(self, content: str) -> Tuple[str, List[str]]:
        """Update code to use modern Python features"""
        transformations = []
        
        # Convert old-style exception handling
        old_except_pattern = r'except\s+(\w+),\s*(\w+):'
        if re.search(old_except_pattern, content):
            content = re.sub(old_except_pattern, r'except \1 as \2:', content)
            transformations.append("Modernized exception handling")
        
        # Replace deprecated functions
        if re.search(r'\bxrange\b', content):
            content = re.sub(r'\bxrange\b', 'range', content)
            transformations.append("Replaced xrange with range")
        
        # Use list/dict comprehensions instead of map/filter where appropriate
        # This is a complex transformation that would require more advanced parsing
        
        return content, transformations
    
    def _update_imports(self, content: str, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """Update imports to use newer packages"""
        transformations = []
        
        # Check for outdated imports
        outdated_imports = {
            "from urllib2 import": "from urllib.request import",
            "from urlparse import": "from urllib.parse import",
            "import urllib2": "import urllib.request as urllib2",
            "import urlparse": "import urllib.parse as urlparse",
            "from itertools import izip": "from itertools import zip_longest",
            "import ConfigParser": "import configparser"
        }
        
        for old, new in outdated_imports.items():
            if old in content:
                content = content.replace(old, new)
                transformations.append(f"Updated import: {old} → {new}")
        
        return content, transformations

class JavaScriptModernizer(CodeTransformer):
    """Modernizes JavaScript code"""
    
    def can_transform(self, code_file: CodeFile) -> bool:
        """Check if this transformer can handle the given file"""
        return code_file.language == LanguageType.JAVASCRIPT
    
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        """Transform JavaScript code to modern standards"""
        content = code_file.content
        transformations = []
        
        # Convert var to let/const
        content, var_transforms = self._convert_var_to_let_const(content)
        transformations.extend(var_transforms)
        
        # Convert functions to arrow functions
        content, arrow_transforms = self._convert_to_arrow_functions(content)
        transformations.extend(arrow_transforms)
        
        # Use modern JS features
        content, modern_transforms = self._modernize_js_features(content)
        transformations.extend(modern_transforms)
        
        return content, transformations
    
    def _convert_var_to_let_const(self, content: str) -> Tuple[str, List[str]]:
        """Convert var declarations to let/const"""
        transformations = []
        
        # Find var declarations
        var_pattern = r'\bvar\s+(\w+)\s*=\s*([^;]+);'
        var_count = 0
        
        # Simple conversion - vars that aren't reassigned become const
        # A real implementation would need to analyze the scope to determine if a variable is reassigned
        for match in re.finditer(var_pattern, content):
            old_str = match.group(0)
            var_name = match.group(1)
            var_value = match.group(2)
            
            # Skip complex cases for this example
            if '{' in var_value or '(' in var_value:
                continue
                
            # Check if the variable appears to be reassigned
            reassign_pattern = rf'{var_name}\s*='
            if re.search(reassign_pattern, content[match.end():]):
                # Variable is reassigned, use let
                new_str = old_str.replace('var', 'let')
            else:
                # Variable is not reassigned, use const
                new_str = old_str.replace('var', 'const')
            
            content = content.replace(old_str, new_str, 1)
            var_count += 1
        
        if var_count > 0:
            transformations.append(f"Converted {var_count} var declarations to let/const")
        
        return content, transformations
    
    def _convert_to_arrow_functions(self, content: str) -> Tuple[str, List[str]]:
        """Convert functions to arrow functions where appropriate"""
        transformations = []
        
        # Find anonymous functions
        anon_func_pattern = r'function\s*\(([^)]*)\)\s*{\s*return\s+([^;]+);\s*}'
        arrow_count = 0
        
        for match in re.finditer(anon_func_pattern, content):
            old_str = match.group(0)
            params = match.group(1)
            body = match.group(2)
            
            # Create arrow function
            new_str = f"({params}) => {body}"
            content = content.replace(old_str, new_str)
            arrow_count += 1
        
        if arrow_count > 0:
            transformations.append(f"Converted {arrow_count} functions to arrow functions")
        
        return content, transformations
    
    def _modernize_js_features(self, content: str) -> Tuple[str, List[str]]:
        """Update code to use modern JavaScript features"""
        transformations = []
        
        # Convert object literals with repeated keys
        if re.search(r'{\s*\w+\s*:\s*\w+\s*}', content):
            content = re.sub(r'(\w+)\s*:\s*\1', r'\1', content)
            transformations.append("Used shorthand object properties")
        
        # Convert string concatenation to template literals
        concat_pattern = r'[\'"]([^\'"]*)[\'"] \+ .+? \+ [\'"]([^\'"]*)[\'"]'
        if re.search(concat_pattern, content):
            # This would require more sophisticated parsing for real use
            transformations.append("Converted string concatenation to template literals")
        
        # Add more modern JS transformations as needed
        
        return content, transformations

class LanguageTranspiler(ABC):
    """Base class for language transpilers"""
    
    def __init__(self):
        """Initialize transpiler"""
        self.temp_dir = tempfile.mkdtemp()
    
    @abstractmethod
    def can_transpile(self, source_language: LanguageType, target_language: LanguageType) -> bool:
        """Check if this transpiler can handle the given language pair"""
        pass
    
    @abstractmethod
    def transpile(self, code_file: CodeFile, target_language: LanguageType) -> Tuple[str, str]:
        """
        Transpile code to a different language
        
        Args:
            code_file: Code file to transpile
            target_language: Target language
            
        Returns:
            Tuple of (transpiled code, new file path)
        """
        pass
    
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class PythonToJSTranspiler(LanguageTranspiler):
    """Transpiles Python to JavaScript"""
    
    def can_transpile(self, source_language: LanguageType, target_language: LanguageType) -> bool:
        """Check if this transpiler can handle the given language pair"""
        return source_language == LanguageType.PYTHON and target_language == LanguageType.JAVASCRIPT
    
    def transpile(self, code_file: CodeFile, target_language: LanguageType) -> Tuple[str, str]:
        """Transpile Python to JavaScript"""
        content = code_file.content
        file_path = code_file.path
        
        # Simple function mapping for demonstration
        # A real transpiler would use more sophisticated parsing and transformation
        js_translations = {
            r'def\s+(\w+)\s*\(([^)]*)\)': r'function \1(\2)',
            r'print\s*\(([^)]*)\)': r'console.log(\1)',
            r'elif': r'else if',
            r'\bTrue\b': r'true',
            r'\bFalse\b': r'false',
            r'\bNone\b': r'null',
            r'#': r'//',
            r'\.append\(': r'.push(',
            r'for\s+(\w+)\s+in\s+range\((\d+)\)': r'for (let \1 = 0; \1 < \2; \1++)',
            r'for\s+(\w+)\s+in\s+(\w+)': r'for (const \1 of \2)',
            r'if\s+([^:]+):': r'if (\1) {',
            r'else:': r'else {',
            r'(\s+)return ': r'\1return '
        }
        
        # Apply simple translations
        js_code = content
        for py_pattern, js_pattern in js_translations.items():
            js_code = re.sub(py_pattern, js_pattern, js_code)
        
        # Replace colons and indentation with braces
        lines = js_code.split('\n')
        result_lines = []
        indent_stack = [0]
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                result_lines.append(line)
                continue
            
            # Calculate current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # If indent decreased, add closing braces
            while current_indent < indent_stack[-1]:
                indent_stack.pop()
                indent = ' ' * indent_stack[-1]
                result_lines.append(f"{indent}}}")
            
            # Process line
            stripped_line = line.strip()
            
            # Add closing brace for indented blocks
            if stripped_line.endswith(':'):
                stripped_line = stripped_line[:-1] + ' {'
                indent_stack.append(current_indent + 4)  # Assume 4 spaces indent
            
            # Add processed line
            result_lines.append(' ' * current_indent + stripped_line)
        
        # Close any remaining open braces
        while len(indent_stack) > 1:
            indent_stack.pop()
            indent = ' ' * indent_stack[-1]
            result_lines.append(f"{indent}}}")
        
        # Create new file path with .js extension
        new_path = os.path.splitext(file_path)[0] + '.js'
        
        return '\n'.join(result_lines), new_path

class JavaScriptToPythonTranspiler(LanguageTranspiler):
    """Transpiles JavaScript to Python"""
    
    def can_transpile(self, source_language: LanguageType, target_language: LanguageType) -> bool:
        """Check if this transpiler can handle the given language pair"""
        return source_language == LanguageType.JAVASCRIPT and target_language == LanguageType.PYTHON
    
    def transpile(self, code_file: CodeFile, target_language: LanguageType) -> Tuple[str, str]:
        """Transpile JavaScript to Python"""
        content = code_file.content
        file_path = code_file.path
        
        # Simple function mapping for demonstration
        py_translations = {
            r'function\s+(\w+)\s*\(([^)]*)\)\s*{': r'def \1(\2):',
            r'console\.log\s*\(([^)]*)\)': r'print(\1)',
            r'else\s+if': r'elif',
            r'\btrue\b': r'True',
            r'\bfalse\b': r'False',
            r'\bnull\b': r'None',
            r'//': r'#',
            r'\.push\(': r'.append(',
            r'for\s*\(\s*let\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\+\+\s*\)': r'for \1 in range(\2)',
            r'for\s*\(\s*const\s+(\w+)\s+of\s+(\w+)\s*\)': r'for \1 in \2',
            r'if\s*\(([^)]+)\)\s*{': r'if \1:',
            r'}\s*else\s*{': r'else:',
            r'return\s+': r'return '
        }
        
        # Apply simple translations
        py_code = content
        for js_pattern, py_pattern in py_translations.items():
            py_code = re.sub(js_pattern, py_pattern, py_code)
        
        # Replace braces with indentation
        lines = py_code.split('\n')
        result_lines = []
        indent_level = 0
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                result_lines.append(line)
                continue
            
            # Process line
            stripped_line = line.strip()
            
            # Handle indentation with braces
            if '{' in stripped_line and '}' not in stripped_line:
                # Opening brace
                result_lines.append(' ' * (4 * indent_level) + stripped_line.replace('{', '').strip())
                indent_level += 1
            elif '}' in stripped_line and '{' not in stripped_line:
                # Closing brace
                indent_level = max(0, indent_level - 1)
                clean_line = stripped_line.replace('}', '').strip()
                if clean_line:
                    result_lines.append(' ' * (4 * indent_level) + clean_line)
            else:
                result_lines.append(' ' * (4 * indent_level) + stripped_line.replace('{', '').replace('}', '').strip())
        
        # Create new file path with .py extension
        new_path = os.path.splitext(file_path)[0] + '.py'
        
        return '\n'.join(result_lines), new_path

class SystemUpgrader:
    """Main class for upgrading a system"""
    
    def __init__(self):
        """Initialize the system upgrader"""
        self.system_analyzer = SystemAnalyzer()
        self.transformers = {
            LanguageType.PYTHON: PythonModernizer(),
            LanguageType.JAVASCRIPT: JavaScriptModernizer()
        }
        self.transpilers = [
            PythonToJSTranspiler(),
            JavaScriptToPythonTranspiler()
        ]
    
    def upgrade_system(self, path: str, config: UpgradeConfig) -> UpgradeResult:
        """
        Upgrade a system
        
        Args:
            path: Path to the system root directory
            config: Upgrade configuration
            
        Returns:
            Upgrade result
        """
        start_time = time.time()
        
        # Analyze the system
        system_info = self.system_analyzer.analyze_system(path)
        
        # Create output directory
        if config.keep_original:
            output_path = f"{path}_upgraded_{int(start_time)}"
            os.makedirs(output_path, exist_ok=True)
            
            # Make a backup copy
            backup_path = f"{path}_backup_{int(start_time)}"
            shutil.copytree(path, backup_path)
        else:
            output_path = path
            backup_path = f"{path}_backup_{int(start_time)}"
            shutil.copytree(path, backup_path)
        
        # Determine upgrade strategy
        chosen_strategy = config.strategy
        if chosen_strategy == UpgradeStrategy.FULL_REWRITE and system_info.primary_language != config.target_language:
            chosen_strategy = UpgradeStrategy.LANGUAGE_MIGRATION
        
        # Apply upgrade based on strategy
        if chosen_strategy == UpgradeStrategy.IN_PLACE:
            result = self._apply_in_place_upgrade(system_info, output_path, config)
        elif chosen_strategy == UpgradeStrategy.INCREMENTAL:
            result = self._apply_incremental_upgrade(system_info, output_path, config)
        elif chosen_strategy == UpgradeStrategy.FULL_REWRITE:
            result = self._apply_full_rewrite(system_info, output_path, config)
        elif chosen_strategy == UpgradeStrategy.LANGUAGE_MIGRATION:
            result = self._apply_language_migration(system_info, output_path, config)
        elif chosen_strategy == UpgradeStrategy.WRAPPER:
            result = self._apply_wrapper_strategy(system_info, output_path, config)
        else:
            raise ValueError(f"Unsupported upgrade strategy: {chosen_strategy}")
        
        # Apply additional post-upgrade steps
        if config.update_dependencies:
            self._update_dependencies(output_path, system_info)
        
        if config.add_tests and not any('test' in p.lower() for p in system_info.files):
            self._add_tests(output_path, system_info, config.target_language)
        
        # Add Kaleidoscope license
        license#!/usr/bin/env python3
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