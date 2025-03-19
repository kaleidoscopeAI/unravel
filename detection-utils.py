"""
File type detection utilities for Kaleidoscope AI
"""

import os
import subprocess
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class FileType(Enum):
    """Enum representing different file types for processing"""
    BINARY = "binary"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    JAVA = "java"
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

class FileDetector:
    """Detects file types based on extension and content"""
    
    def __init__(self):
        """Initialize the file detector"""
        self.ext_map = {
            ".exe": FileType.BINARY,
            ".dll": FileType.BINARY,
            ".so": FileType.BINARY,
            ".dylib": FileType.BINARY,
            ".js": FileType.JAVASCRIPT,
            ".mjs": FileType.JAVASCRIPT,
            ".py": FileType.PYTHON,
            ".cpp": FileType.CPP,
            ".cc": FileType.CPP,
            ".c": FileType.C,
            ".cs": FileType.CSHARP,
            ".java": FileType.JAVA,
            ".asm": FileType.ASSEMBLY,
            ".s": FileType.ASSEMBLY
        }
    
    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect the type of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileType: The detected file type
        """
        # First check by extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in self.ext_map:
            return self.ext_map[file_ext]
        
        # If extension doesn't give us enough info, try to use 'file' command
        try:
            file_output = subprocess.check_output(
                ["file", file_path], 
                universal_newlines=True
            )
            
            # Check content markers
            if "ELF" in file_output or "PE32" in file_output or "Mach-O" in file_output:
                return FileType.BINARY
            elif "JavaScript" in file_output:
                return FileType.JAVASCRIPT
            elif "Python" in file_output:
                return FileType.PYTHON
            elif "C++ source" in file_output:
                return FileType.CPP
            elif "C source" in file_output:
                return FileType.C
            elif "C# source" in file_output:
                return FileType.CSHARP
            elif "Java source" in file_output:
                return FileType.JAVA
            elif "assembler source" in file_output:
                return FileType.ASSEMBLY
                
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Could not use 'file' command: {str(e)}")
            
        # If we can't determine the type, check file content directly
        try:
            # Read first 1000 bytes of the file
            with open(file_path, 'rb') as f:
                content = f.read(1000)
                
            # Check binary markers
            if b'\x00' in content[:100]:
                return FileType.BINARY
                
            # Check text markers for specific languages
            text_content = content.decode('utf-8', errors='ignore').lower()
            
            if "function" in text_content or "const" in text_content or "var" in text_content or "let" in text_content:
                return FileType.JAVASCRIPT
            elif "def " in text_content or "class " in text_content or "import " in text_content or "from " in text_content:
                return FileType.PYTHON
            elif "#include" in text_content:
                if "class" in text_content or "template" in text_content:
                    return FileType.CPP
                return FileType.C
            elif "public class" in text_content or "private class" in text_content:
                if "System.Collections" in text_content:
                    return FileType.CSHARP
                return FileType.JAVA
                
        except Exception as e:
            logger.warning(f"Error reading file content: {str(e)}")
        
        # Default to unknown if we can't determine
        return FileType.UNKNOWN
