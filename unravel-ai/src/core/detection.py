"""
Unravel AI - File Type Detection Module
Detects file types and programming languages
"""

import os
import re
import magic
import logging
import subprocess
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class FileType(Enum):
    """File types supported by Unravel AI"""
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
    SWIFT = "swift"
    KOTLIN = "kotlin"
    PHP = "php"
    RUBY = "ruby"
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

class FileDetector:
    """Detects file types and programming languages"""
    
    def __init__(self):
        """Initialize the file detector"""
        self.mime = magic.Magic(mime=True)
        
        # File extension mappings
        self.ext_map = {
            # Binaries
            ".exe": FileType.BINARY,
            ".dll": FileType.BINARY,
            ".so": FileType.BINARY,
            ".dylib": FileType.BINARY,
            ".bin": FileType.BINARY,
            ".o": FileType.BINARY,
            ".obj": FileType.BINARY,
            
            # Scripts and source files
            ".js": FileType.JAVASCRIPT,
            ".mjs": FileType.JAVASCRIPT,
            ".ts": FileType.TYPESCRIPT,
            ".tsx": FileType.TYPESCRIPT,
            ".jsx": FileType.JAVASCRIPT,
            ".py": FileType.PYTHON,
            ".cpp": FileType.CPP,
            ".cc": FileType.CPP,
            ".cxx": FileType.CPP,
            ".c": FileType.C,
            ".h": FileType.C,
            ".hpp": FileType.CPP,
            ".cs": FileType.CSHARP,
            ".java": FileType.JAVA,
            ".go": FileType.GO,
            ".rs": FileType.RUST,
            ".swift": FileType.SWIFT,
            ".kt": FileType.KOTLIN,
            ".php": FileType.PHP,
            ".rb": FileType.RUBY,
            ".asm": FileType.ASSEMBLY,
            ".s": FileType.ASSEMBLY
        }
        
        # MIME type mappings
        self.mime_map = {
            "application/x-executable": FileType.BINARY,
            "application/x-sharedlib": FileType.BINARY,
            "application/x-mach-binary": FileType.BINARY,
            "application/x-dosexec": FileType.BINARY,
            "application/x-object": FileType.BINARY,
            
            "text/javascript": FileType.JAVASCRIPT,
            "application/javascript": FileType.JAVASCRIPT,
            "text/x-python": FileType.PYTHON,
            "text/x-c": FileType.C,
            "text/x-c++": FileType.CPP,
            "text/x-java": FileType.JAVA,
            "text/x-csharp": FileType.CSHARP,
            "text/x-go": FileType.GO,
            "text/x-rust": FileType.RUST,
            "text/x-swift": FileType.SWIFT,
            "text/x-kotlin": FileType.KOTLIN,
            "text/x-php": FileType.PHP,
            "text/x-ruby": FileType.RUBY,
            "text/x-asm": FileType.ASSEMBLY
        }
    
    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect the type of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        # Check file extension first
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.ext_map:
            return self.ext_map[ext]
        
        # Try to detect by MIME type
        try:
            mime_type = self.mime.from_file(file_path)
            
            # Look for known MIME types
            for mime_pattern, file_type in self.mime_map.items():
                if mime_pattern in mime_type:
                    return file_type
            
            # Check if binary or text
            if "text/" in mime_type:
                # Try to infer from content
                return self._infer_from_content(file_path)
            elif "application/" in mime_type or "binary" in mime_type:
                return FileType.BINARY
        except Exception as e:
            logger.error(f"Error detecting MIME type: {str(e)}")
        
        # Try using the 'file' command
        try:
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            # Check output for clues
            lower_output = output.lower()
            
            if any(x in lower_output for x in ["elf", "executable", "binary", "mach-o", "pe32"]):
                return FileType.BINARY
            elif "javascript" in lower_output:
                return FileType.JAVASCRIPT
            elif "python" in lower_output:
                return FileType.PYTHON
            elif "c++ source" in lower_output:
                return FileType.CPP
            elif "c source" in lower_output:
                return FileType.C
            elif "java source" in lower_output:
                return FileType.JAVA
            elif "c#" in lower_output:
                return FileType.CSHARP
            elif "go source" in lower_output:
                return FileType.GO
            elif "rust" in lower_output:
                return FileType.RUST
            elif "swift source" in lower_output:
                return FileType.SWIFT
            elif "php script" in lower_output:
                return FileType.PHP
            elif "ruby script" in lower_output:
                return FileType.RUBY
            elif "assembler source" in lower_output:
                return FileType.ASSEMBLY
        except Exception as e:
            logger.error(f"Error running 'file' command: {str(e)}")
        
        # If all else fails, try to infer from content
        return self._infer_from_content(file_path)
    
    def _infer_from_content(self, file_path: str) -> FileType:
        """
        Infer file type from content
        
        Args:
            file_path: Path to the file
            
        Returns:
            Inferred file type
        """
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(4096)  # Read first 4K of the file
                
                # Look for language indicators
                if re.search(r'import\s+{.*?}\s+from|require\(|export\s+default|=>|function\s+\w+\s*\(', content):
                    return FileType.JAVASCRIPT
                elif re.search(r'interface\s+\w+|class\s+\w+\s+implements|<\w+>|:.*?;', content):
                    return FileType.TYPESCRIPT
                elif re.search(r'import\s+\w+|def\s+\w+\s*\(.*\):|class\s+\w+\s*:', content):
                    return FileType.PYTHON
                elif re.search(r'#include\s+<\w+\.h>|template\s+<typename|std::', content):
                    return FileType.CPP
                elif re.search(r'#include\s+<\w+\.h>|void\s+\w+\s*\(|int\s+main\s*\(', content):
                    return FileType.C
                elif re.search(r'public\s+class|public\s+static\s+void\s+main|@Override', content):
                    return FileType.JAVA
                elif re.search(r'namespace\s+\w+|public\s+class\s+\w+\s*:|using\s+System;', content):
                    return FileType.CSHARP
                elif re.search(r'package\s+main|func\s+\w+\s*\(|import\s+\(', content):
                    return FileType.GO
                elif re.search(r'fn\s+\w+\s*\(|impl\s+\w+|use\s+std::', content):
                    return FileType.RUST
                elif re.search(r'import\s+Foundation|@objc|class\s+\w+\s*:\s*\w+', content):
                    return FileType.SWIFT
                elif re.search(r'<?php|namespace\s+\w+;|\$\w+\s*=', content):
                    return FileType.PHP
                elif re.search(r'require\s+[\'"]\w+[\'"]|def\s+\w+\s*(\(|$)|class\s+\w+\s*<', content):
                    return FileType.RUBY
                elif re.search(r'\.globl|\.section|\.text', content):
                    return FileType.ASSEMBLY
        except Exception as e:
            logger.error(f"Error inferring file type from content: {str(e)}")
        
        # Default to unknown
        return FileType.UNKNOWN
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a file and return detailed information
        
        Args:
            file_path: Path to the file
            
        Returns:
            File analysis information
        """
        file_type = self.detect_file_type(file_path)
        
        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": file_type.value,
            "file_size": os.path.getsize(file_path),
            "mime_type": self.mime.from_file(file_path),
            "last_modified": os.path.getmtime(file_path)
        }
        
        # Add file-type specific analysis
        if file_type == FileType.BINARY:
            result.update(self._analyze_binary(file_path))
        elif file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT]:
            result.update(self._analyze_javascript(file_path))
        elif file_type == FileType.PYTHON:
            result.update(self._analyze_python(file_path))
        elif file_type in [FileType.C, FileType.CPP]:
            result.update(self._analyze_c_cpp(file_path))
        
        return result
    
    def _analyze_binary(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze binary file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary analysis info
        """
        result = {
            "is_executable": os.access(file_path, os.X_OK),
            "architecture": "unknown",
            "format": "unknown"
        }
        
        try:
            # Try to get more info using 'file' command
            output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            # Parse architecture and format
            if "ELF" in output:
                result["format"] = "ELF"
                if "x86-64" in output:
                    result["architecture"] = "x86_64"
                elif "80386" in output:
                    result["architecture"] = "x86"
                elif "ARM" in output:
                    result["architecture"] = "ARM"
            elif "PE32" in output:
                result["format"] = "PE"
                if "x86-64" in output:
                    result["architecture"] = "x86_64"
                else:
                    result["architecture"] = "x86"
            elif "Mach-O" in output:
                result["format"] = "Mach-O"
                if "x86_64" in output:
                    result["architecture"] = "x86_64"
                elif "arm64" in output:
                    result["architecture"] = "ARM64"
        except Exception as e:
            logger.error(f"Error analyzing binary: {str(e)}")
        
        return result
    
    def _analyze_javascript(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript file
        
        Args:
            file_path: Path to the file
            
        Returns:
            JavaScript analysis info
        """
        result = {
            "imports": [],
            "exports": [],
            "functions": [],
            "classes": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract imports
                import_patterns = [
                    r'import\s+{(.*?)}\s+from\s+[\'"](.+?)[\'"]',
                    r'import\s+(\w+)\s+from\s+[\'"](.+?)[\'"]',
                    r'require\s*\(\s*[\'"](.+?)[\'"]\s*\)'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        if len(match.groups()) == 2:
                            result["imports"].append({
                                "imported": match.group(1),
                                "source": match.group(2)
                            })
                        elif len(match.groups()) == 1:
                            result["imports"].append({
                                "source": match.group(1)
                            })
                
                # Extract functions
                for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:function|\(.*?\)\s*=>)|\()', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
                # Extract exports
                for match in re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', content):
                    result["exports"].append(match.group(1))
                
        except Exception as e:
            logger.error(f"Error analyzing JavaScript: {str(e)}")
        
        return result
    
    def _analyze_python(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze Python file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Python analysis info
        """
        result = {
            "imports": [],
            "functions": [],
            "classes": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract imports
                import_patterns = [
                    r'import\s+(\w+)',
                    r'from\s+(\w+(?:\.\w+)*)\s+import\s+(.*)'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        if len(match.groups()) == 1:
                            result["imports"].append({
                                "module": match.group(1)
                            })
                        elif len(match.groups()) == 2:
                            result["imports"].append({
                                "module": match.group(1),
                                "imports": [x.strip() for x in match.group(2).split(',')]
                            })
                
                # Extract functions
                for match in re.finditer(r'def\s+(\w+)\s*\(', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
        except Exception as e:
            logger.error(f"Error analyzing Python: {str(e)}")
        
        return result
    
    def _analyze_c_cpp(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze C/C++ file
        
        Args:
            file_path: Path to the file
            
        Returns:
            C/C++ analysis info
        """
        result = {
            "includes": [],
            "functions": [],
            "classes": [],
            "preprocessor": []
        }
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Extract includes
                for match in re.finditer(r'#include\s+[<"](.+?)[>"]', content):
                    result["includes"].append(match.group(1))
                
                # Extract functions
                for match in re.finditer(r'(?:int|void|bool|char|double|float|auto|std::string|string)\s+(\w+)\s*\(', content):
                    result["functions"].append(match.group(1))
                
                # Extract classes
                for match in re.finditer(r'class\s+(\w+)', content):
                    result["classes"].append(match.group(1))
                
                # Extract preprocessor directives
                for match in re.finditer(r'#define\s+(\w+)', content):
                    result["preprocessor"].append({
                        "type": "define",
                        "name": match.group(1)
                    })
                
        except Exception as e:
            logger.error(f"Error analyzing C/C++: {str(e)}")
        
        return result
