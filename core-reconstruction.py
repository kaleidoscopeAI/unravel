#!/usr/bin/env python3
"""
Unravel AI - Software Reconstruction Engine
Generates improved versions of ingested software
"""

import os
import re
import ast
import logging
import asyncio
import tempfile
import shutil
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from src.utils.config import config
from src.core.llm import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionConfig:
    """Configuration for software reconstruction"""
    quality_level: str = "high"  # low, medium, high
    add_comments: bool = True
    improve_security: bool = True
    optimize_performance: bool = True
    modernize_codebase: bool = True
    add_testing: bool = False
    target_language: Optional[str] = None  # For language translation
    extra_instructions: Optional[str] = None
    
    # Advanced options
    preserve_functionality: bool = True
    rename_variables: bool = False
    code_style: str = "clean"  # clean, compact, verbose
    custom_patterns: List[Dict[str, str]] = field(default_factory=list)

class ReconstructionEngine:
    """Engine for reconstructing and improving software"""
    
    def __init__(self, output_dir: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        """Initialize the reconstruction engine"""
        self.output_dir = output_dir or os.path.join(config.get("WORK_DIR", "./workdir"), "reconstructed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create or use the provided LLM client
        self.llm_client = llm_client or LLMClient(
            api_key=config.get("LLM_API_KEY"),
            model=config.get("LLM_MODEL"),
            endpoint=config.get("LLM_ENDPOINT")
        )
        
        # Languages we can handle
        self.supported_languages = {
            "python": [".py"],
            "javascript": [".js", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"]
        }
        
        # Language pattern matchers
        self.language_matchers = {
            "python": re.compile(r'import\s+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+\s*:'),
            "javascript": re.compile(r'function\s+\w+\s*\(|const\s+\w+\s*=|import\s+.*from|export\s+'),
            "typescript": re.compile(r'interface\s+\w+|type\s+\w+\s*=|class\s+\w+\s+implements'),
            "java": re.compile(r'public\s+class|private\s+class|package\s+\w+;|import\s+\w+\.'),
            "c": re.compile(r'#include\s+[<"]|void\s+\w+\s*\(|int\s+main\s*\('),
            "cpp": re.compile(r'#include\s+[<"]|\w+::\w+|namespace\s+\w+|template\s*<'),
            "csharp": re.compile(r'namespace\s+\w+|using\s+\w+;|public\s+class|private\s+class'),
            "go": re.compile(r'package\s+main|import\s+\(|func\s+\w+\s*\('),
            "rust": re.compile(r'fn\s+\w+\s*\(|impl\s+\w+|use\s+\w+::'),
            "ruby": re.compile(r'require\s+[\'"]|def\s+\w+|class\s+\w+\s*<')
        }
        
        # Code transformers
        self.transformers = {
            "python": self._transform_python,
            "javascript": self._transform_javascript,
            "typescript": self._transform_typescript,
            "java": self._transform_java,
            "c": self._transform_c,
            "cpp": self._transform_cpp,
            "csharp": self._transform_csharp,
            "go": self._transform_go,
            "rust": self._transform_rust,
            "ruby": self._transform_ruby
        }
    
    async def reconstruct_file(self, 
                             file_path: str, 
                             config: Optional[ReconstructionConfig] = None, 
                             output_path: Optional[str] = None) -> str:
        """Reconstruct a single file"""
        config = config or ReconstructionConfig()
        
        try:
            # Read the file
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Detect language
            ext = os.path.splitext(file_path)[1].lower()
            language = None
            
            # Try by extension first
            for lang, extensions in self.supported_languages.items():
                if ext in extensions:
                    language = lang
                    break
            
            # If extension didn't work, try content matching
            if not language:
                for lang, pattern in self.language_matchers.items():
                    if pattern.search(content):
                        language = lang
                        break
            
            if not language:
                language = "unknown"
                logger.warning(f"Could not detect language for {file_path}, using default reconstruction")
            
            # Apply transformations
            transform_func = self.transformers.get(language, self._transform_generic)
            improved_content = await transform_func(content, config)
            
            # Write the reconstructed file
            if not output_path:
                filename = os.path.basename(file_path)
                reconstruct_id = str(uuid.uuid4())[:8]
                output_dir = os.path.join(self.output_dir, reconstruct_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"reconstructed_{filename}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(improved_content)
            
            logger.info(f"Reconstructed {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.exception(f"Error reconstructing {file_path}: {str(e)}")
            raise
    
    async def reconstruct_directory(self, 
                                  directory_path: str, 
                                  config: Optional[ReconstructionConfig] = None, 
                                  output_dir: Optional[str] = None) -> List[str]:
        """Reconstruct all files in a directory"""
        config = config or ReconstructionConfig()
        
        # Create output directory
        if not output_dir:
            reconstruct_id = str(uuid.uuid4())[:8]
            output_dir = os.path.join(self.output_dir, reconstruct_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        reconstructed_files = []
        
        # Get all files
        files_to_process = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory_path)
                output_path = os.path.join(output_dir, rel_path)
                
                # Skip .git and other hidden directories
                if any(part.startswith('.') for part in rel_path.split(os.sep)):
                    continue
                
                # Skip common non-code files
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', 
                               '.woff', '.ttf', '.eot', '.woff2', '.mp3', '.mp4', 
                               '.zip', '.tar.gz', '.pdf', '.doc', '.docx')):
                    # Just copy these files
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(file_path, output_path)
                    reconstructed_files.append(output_path)
                    continue
                
                # Determine if this is likely a code file
                ext = os.path.splitext(file)[1].lower()
                is_code_file = False
                
                for exts in self.supported_languages.values():
                    if ext in exts:
                        is_code_file = True
                        break
                
                if not is_code_file and not file.endswith(('.json', '.xml', '.yml', '.yaml', '.md', '.txt')):
                    # Just copy non-code files
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(file_path, output_path)
                    reconstructed_files.append(output_path)
                    continue
                
                files_to_process.append((file_path, output_path))
        
        # Process all files (with concurrency limits)
        tasks = []
        for file_path, output_path in files_to_process:
            task = asyncio.create_task(self.reconstruct_file(file_path, config, output_path))
            tasks.append(task)
            
            # Only run 5 concurrent decompilations
            if len(tasks) >= 5:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                reconstructed_files.extend([t.result() for t in done if not t.exception()])
        
        # Wait for remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks)
            reconstructed_files.extend([t.result() for t in done if not t.exception()])
        
        return reconstructed_files
    
    async def _transform_generic(self, content: str, config: ReconstructionConfig) -> str:
        """Generic transformation using LLM"""
        # Truncate content if it's too long
        if len(content) > 12000:
            logger.warning(f"Content too long ({len(content)} chars), truncating to 12000 chars")
            content = content[:12000]
        
        # Build instruction prompt
        instructions = [
            "Improve this code with the following enhancements:",
            "1. Better organization and structure"
        ]
        
        if config.add_comments:
            instructions.append("2. Add meaningful comments")
        
        if config.improve_security:
            instructions.append("3. Fix security issues")
        
        if config.optimize_performance:
            instructions.append("4. Optimize for performance")
        
        if config.modernize_codebase:
            instructions.append("5. Modernize the code using current best practices")
        
        if config.add_testing:
            instructions.append("6. Add appropriate tests")
        
        if config.extra_instructions:
            instructions.append(f"7. {config.extra_instructions}")
        
        # Different quality levels
        if config.quality_level == "low":
            system_message = "You are a practical software engineer. Focus on minimal, functional improvements."
        elif config.quality_level == "medium":
            system_message = "You are a professional software engineer. Balance improvements with practicality."
        else:  # high
            system_message = "You are an expert software engineer. Create elegant, maintainable, and efficient code."
        
        # Create prompt
        prompt = f"""
        {system_message}
        
        I need you to improve the following code:
        
        ```
        {content}
        ```
        
        {chr(10).join(instructions)}
        
        Provide ONLY the improved code without explanations.
        """
        
        # Get improved code from LLM
        improved_content = await self.llm_client.complete(prompt, system_message=system_message)
        
        # Extract code from markdown code blocks if present
        if "```" in improved_content:
            matches = re.findall(r'```(?:\w*\n)?(.*?)```', improved_content, re.DOTALL)
            if matches:
                improved_content = matches[0]
        
        return improved_content
    
    async def _transform_python(self, content: str, config: ReconstructionConfig) -> str:
        """Transform Python code"""
        try:
            # For short files, try to use AST parsing and transformation
            if len(content) < 5000 and config.quality_level != "low":
                try:
                    tree = ast.parse(content)
                    
                    # TODO: Implement AST-based transformations
                    # For now, fall back to LLM-based transformation
                    
                except SyntaxError:
                    logger.warning("Python syntax error, falling back to LLM transformation")
                    
            # Use LLM for transformation
            system_message = """You are a Python expert specializing in clean, Pythonic code. 
            Focus on idiomatic Python patterns, type annotations, docstrings, and following PEP 8."""
            
            return await self._transform_generic(content, config)
            
        except Exception as e:
            logger.exception(f"Error transforming Python code: {str(e)}")
            # Fall back to original content
            return content
    
    async def _transform_javascript(self, content: str, config: ReconstructionConfig) -> str:
        """Transform JavaScript code"""
        system_message = """You are a JavaScript expert. 
        Focus on modern ES6+ features, functional programming patterns, and clean code principles."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_typescript(self, content: str, config: ReconstructionConfig) -> str:
        """Transform TypeScript code"""
        system_message = """You are a TypeScript expert. 
        Focus on strong typing, interfaces, and modern TypeScript patterns."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_java(self, content: str, config: ReconstructionConfig) -> str:
        """Transform Java code"""
        system_message = """You are a Java expert.
        Focus on clean object-oriented design, appropriate design patterns, and Java best practices."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_c(self, content: str, config: ReconstructionConfig) -> str:
        """Transform C code"""
        system_message = """You are a C language expert.
        Focus on memory safety, efficient algorithms, and clear structured code."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_cpp(self, content: str, config: ReconstructionConfig) -> str:
        """Transform C++ code"""
        system_message = """You are a C++ expert.
        Focus on modern C++ features, RAII principles, and efficient memory management."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_csharp(self, content: str, config: ReconstructionConfig) -> str:
        """Transform C# code"""
        system_message = """You are a C# expert.
        Focus on .NET best practices, LINQ, async/await patterns, and clean OOP design."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_go(self, content: str, config: ReconstructionConfig) -> str:
        """Transform Go code"""
        system_message = """You are a Go expert.
        Focus on idiomatic Go patterns, error handling, and concurrency best practices."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_rust(self, content: str, config: ReconstructionConfig) -> str:
        """Transform Rust code"""
        system_message = """You are a Rust expert.
        Focus on memory safety, ownership model, and performant idiomatic Rust."""
        
        return await self._transform_generic(content, config)
    
    async def _transform_ruby(self, content: str, config: ReconstructionConfig) -> str:
        """Transform Ruby code"""
        system_message = """You are a Ruby expert.
        Focus on Ruby idioms, elegant object-oriented design, and maintainable code."""
        
        return await self._transform_generic(content, config)
    
    async def translate_to_language(self, content: str, source_language: str, target_language: str) -> str:
        """Translate code from one language to another"""
        system_message = f"""You are an expert programmer fluent in both {source_language} and {target_language}.
        Your task is to translate code between these languages while preserving functionality and idioms."""
        
        prompt = f"""
        Translate this {source_language} code to {target_language}:
        
        ```{source_language}
        {content}
        ```
        
        Follow these guidelines:
        1. Preserve all functionality exactly
        2. Use idiomatic {target_language} patterns and best practices
        3. Keep variable names and structure similar when possible
        4. Add comments explaining any non-obvious translations
        
        Return ONLY the translated {target_language} code.
        """
        
        translated = await self.llm_client.complete(prompt, system_message=system_message)
        
        # Extract code from markdown code blocks if present
        if "```" in translated:
            matches = re.findall(r'```(?:\w*\n)?(.*?)```', translated, re.DOTALL)
            if matches:
                translated = matches[0]
        
        return translated