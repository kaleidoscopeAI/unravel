import os
import uuid
import shutil
import json
import asyncio
import logging
import tempfile
import subprocess
import re
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.services.llm_service import llm_service
from app.models.schema import FileType, ProcessStatus

logger = logging.getLogger(__name__)

class KaleidoscopeService:
    """Service implementing the Kaleidoscope functionality"""
    
    def __init__(self):
        """Initialize the Kaleidoscope service"""
        # Create work directories
        self.work_dir = os.path.join(settings.UPLOAD_DIR, "kaleidoscope_workdir")
        self.source_dir = os.path.join(self.work_dir, "source")
        self.decompiled_dir = os.path.join(self.work_dir, "decompiled")
        self.specs_dir = os.path.join(self.work_dir, "specs")
        self.reconstructed_dir = os.path.join(self.work_dir, "reconstructed")
        
        # Create directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.decompiled_dir, exist_ok=True)
        os.makedirs(self.specs_dir, exist_ok=True)
        os.makedirs(self.reconstructed_dir, exist_ok=True)
        
        # Initialize dependency graph
        self.dependency_graph = nx.DiGraph()
    
    async def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect the type of an input file
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileType: The detected file type
        """
        # Check file extension first
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to file types
        ext_map = {
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
        
        if file_ext in ext_map:
            return ext_map[file_ext]
        
        # If extension doesn't match, try to detect file type using file command
        try:
            file_output = await asyncio.create_subprocess_exec(
                "file", file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await file_output.communicate()
            output = stdout.decode()
            
            if any(x in output for x in ["ELF", "PE32", "Mach-O"]):
                return FileType.BINARY
            elif "JavaScript" in output:
                return FileType.JAVASCRIPT
            elif "Python" in output:
                return FileType.PYTHON
            elif "C++ source" in output:
                return FileType.CPP
            elif "C source" in output:
                return FileType.C
            elif "assembler source" in output:
                return FileType.ASSEMBLY
        except:
            logger.warning("Could not use 'file' command to detect file type")
        
        return FileType.UNKNOWN
    
    async def ingest_software(self, file_path: str, software_id: int) -> Dict[str, Any]:
        """
        Main entry point for software ingestion
        
        Args:
            file_path: Path to the software file to ingest
            software_id: ID of the software in the database
            
        Returns:
            Dict: Results of ingestion process
        """
        logger.info(f"Starting ingestion of {file_path} for software ID {software_id}")
        
        # Create a unique working directory for this software
        work_subdir = f"software_{software_id}_{uuid.uuid4().hex[:8]}"
        software_work_dir = os.path.join(self.work_dir, work_subdir)
        os.makedirs(software_work_dir, exist_ok=True)
        
        # Copy source file to work directory
        source_filename = os.path.basename(file_path)
        source_dest = os.path.join(software_work_dir, source_filename)
        shutil.copy2(file_path, source_dest)
        
        # Detect file type
        file_type = await self.detect_file_type(source_dest)
        logger.info(f"Detected file type: {file_type}")
        
        # Set up result dictionary
        result = {
            "software_id": software_id,
            "original_file": file_path,
            "work_file": source_dest,
            "file_type": file_type,
            "work_dir": software_work_dir,
            "decompiled_files": [],
            "spec_files": [],
            "reconstructed_files": [],
            "status": ProcessStatus.PENDING
        }
        
        # Process based on file type
        try:
            if file_type == FileType.BINARY:
                result = await self._process_binary(source_dest, result)
            elif file_type == FileType.JAVASCRIPT:
                result = await self._process_javascript(source_dest, result)
            elif file_type == FileType.PYTHON:
                result = await self._process_python(source_dest, result)
            elif file_type in [FileType.C, FileType.CPP]:
                result = await self._process_c_cpp(source_dest, result)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                result["status"] = ProcessStatus.FAILED
                result["error_message"] = f"Unsupported file type: {file_type}"
                return result
            
            # Generate specifications
            if result.get("decompiled_files"):
                spec_files = await self._generate_specifications(result["decompiled_files"], software_work_dir)
                result["spec_files"] = spec_files
                
                # Reconstruct software
                if spec_files:
                    reconstructed_files = await self._reconstruct_software(spec_files, software_work_dir)
                    result["reconstructed_files"] = reconstructed_files
                    result["status"] = ProcessStatus.COMPLETED
                else:
                    result["status"] = ProcessStatus.FAILED
                    result["error_message"] = "Failed to generate specifications"
            else:
                result["status"] = ProcessStatus.FAILED
                result["error_message"] = "Failed to decompile software"
                
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            result["status"] = ProcessStatus.ERROR
            result["error_message"] = str(e)
            
        return result
    
    async def _process_binary(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a binary file using r2 and other tools"""
        logger.info(f"Processing binary: {file_path}")
        
        decompiled_dir = os.path.join(result["work_dir"], "decompiled")
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Use radare2 for initial analysis
        radare_output = os.path.join(decompiled_dir, "radare_analysis.txt")
        try:
            proc = await asyncio.create_subprocess_exec(
                "r2", "-q", "-c", "aaa; s main; pdf", file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            with open(radare_output, 'w') as f:
                f.write(stdout.decode())
                
            result["decompiled_files"].append(radare_output)
            logger.info(f"Radare2 analysis complete: {radare_output}")
        except Exception as e:
            logger.error(f"Radare2 analysis failed: {str(e)}")
        
        # For binaries we'll use LLM-based analysis instead of actual decompilation
        # as we don't want to require heavy tools in the container
        analysis_output = os.path.join(decompiled_dir, "llm_analysis.txt")
        try:
            # Read the radare2 output if available
            analysis_text = ""
            if os.path.exists(radare_output):
                with open(radare_output, 'r') as f:
                    analysis_text = f.read()
            
            # Use LLM to analyze the binary based on radare2 output
            if analysis_text:
                prompt = f"""
                As an expert reverse engineer, analyze this binary output from radare2:
                
                {analysis_text[:10000]}  # Use first 10k chars to avoid context limit
                
                Provide a detailed analysis of:
                1. The purpose and functionality of this binary
                2. Key functions and their roles
                3. Data structures identified
                4. Control flow and algorithms
                5. Any notable security features or vulnerabilities
                
                Format your response as structured analysis that could be used for reconstruction.
                """
                
                analysis_result = await llm_service.generate_completion(prompt)
                
                with open(analysis_output, 'w') as f:
                    f.write(analysis_result)
                
                result["decompiled_files"].append(analysis_output)
                logger.info(f"LLM analysis complete: {analysis_output}")
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            
        # If we have at least one successful analysis, continue
        if not result["decompiled_files"]:
            logger.error(f"All binary analysis methods failed for {file_path}")
            
        return result
    
    async def _process_javascript(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JavaScript file"""
        logger.info(f"Processing JavaScript: {file_path}")
        
        decompiled_dir = os.path.join(result["work_dir"], "decompiled")
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Beautify JavaScript
        beautified_output = os.path.join(decompiled_dir, "beautified.js")
        try:
            proc = await asyncio.create_subprocess_exec(
                "js-beautify", file_path, "-o", beautified_output,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            
            result["decompiled_files"].append(beautified_output)
            logger.info(f"JavaScript beautification complete: {beautified_output}")
            
            # Extract structure using LLM analysis
            with open(beautified_output, 'r') as f:
                js_content = f.read()
            
            # Use chunks if the file is large
            if len(js_content) > 10000:
                analysis_chunks = []
                for i in range(0, len(js_content), 10000):
                    chunk = js_content[i:i+10000]
                    analysis_chunks.append(chunk)
                
                # Process each chunk with LLM
                all_analyses = []
                for i, chunk in enumerate(analysis_chunks):
                    prompt = f"""
                    As an expert JavaScript developer, analyze this code chunk {i+1} of {len(analysis_chunks)}:
                    
                    ```javascript
                    {chunk}
                    ```
                    
                    Extract:
                    1. Functions and their purpose
                    2. Classes and their methods
                    3. Key data structures
                    4. Dependencies and imports
                    5. Overall architecture and patterns
                    
                    Format your response as JSON with these sections.
                    """
                    
                    analysis = await llm_service.generate_completion(prompt)
                    all_analyses.append(analysis)
                
                # Combine analyses
                combined_analysis = "\n\n".join(all_analyses)
                ast_output = os.path.join(decompiled_dir, "js_analysis.json")
                with open(ast_output, 'w') as f:
                    f.write(combined_analysis)
                
                result["decompiled_files"].append(ast_output)
                logger.info(f"JavaScript analysis complete: {ast_output}")
            else:
                # For smaller files, analyze the whole thing at once
                prompt = f"""
                As an expert JavaScript developer, analyze this code:
                
                ```javascript
                {js_content}
                ```
                
                Extract:
                1. Functions and their purpose
                2. Classes and their methods
                3. Key data structures
                4. Dependencies and imports
                5. Overall architecture and patterns
                
                Format your response as JSON with these sections.
                """
                
                analysis = await llm_service.generate_completion(prompt)
                ast_output = os.path.join(decompiled_dir, "js_analysis.json")
                with open(ast_output, 'w') as f:
                    f.write(analysis)
                
                result["decompiled_files"].append(ast_output)
                logger.info(f"JavaScript analysis complete: {ast_output}")
            
        except Exception as e:
            logger.error(f"JavaScript processing failed: {str(e)}")
            
        return result
    
    async def _process_python(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Python file for analysis"""
        logger.info(f"Processing Python: {file_path}")
        
        decompiled_dir = os.path.join(result["work_dir"], "decompiled")
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Python AST analysis using built-in ast module
        ast_output = os.path.join(decompiled_dir, "python_ast_analysis.json")
        
        # Create a temporary Python script to analyze the file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
            temp.write("""
import ast
import json
import sys

def analyze_python_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
        
        # Extract functions, classes, imports
        functions = []
        classes = []
        imports = []
        global_vars = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', None),
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'lineno': item.lineno,
                        })
                
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'bases': [base.id if isinstance(base, ast.Name) else 'complex_base' for base in node.bases],
                    'lineno': node.lineno,
                })
            elif isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'name': name.name,
                        'asname': name.asname,
                        'lineno': node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    'module': node.module,
                    'names': [{'name': name.name, 'asname': name.asname} for name in node.names],
                    'lineno': node.lineno,
                })
            elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():  # Assume constants are UPPERCASE
                        global_vars.append({
                            'name': target.id,
                            'lineno': node.lineno,
                        })
        
        analysis = {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'global_vars': global_vars,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Python AST analysis complete")
        return True
    except SyntaxError as e:
        print(f"Error parsing Python file: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <python_file> <output_file>")
        sys.exit(1)
    
    success = analyze_python_file(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
            """)
            temp_path = temp.name
        
        try:
            # Run the Python AST analysis
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path, file_path, ast_output,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            
            if os.path.exists(ast_output):
                result["decompiled_files"].append(ast_output)
                logger.info(f"Python AST analysis complete: {ast_output}")
                
                # Also add the original source since Python is already human-readable
                source_copy = os.path.join(decompiled_dir, "source.py")
                shutil.copy2(file_path, source_copy)
                result["decompiled_files"].append(source_copy)
            
        except Exception as e:
            logger.error(f"Python analysis failed: {str(e)}")
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        
        # Also generate an LLM-based analysis
        llm_analysis_output = os.path.join(decompiled_dir, "python_llm_analysis.txt")
        try:
            # Read the Python source
            with open(file_path, 'r') as f:
                python_content = f.read()
            
            # Use LLM to analyze the Python code
            prompt = f"""
            As an expert Python developer, analyze this code:
            
            ```python
            {python_content[:15000]}  # Use first 15k chars to avoid context limit
            ```
            
            Provide:
            1. A summary of what this code does
            2. The key functions and their purposes
            3. Classes and their methods
            4. Dependencies and imports
            5. Overall architecture patterns
            6. Any potential issues or improvements
            
            Format your response as structured text with clear sections.
            """
            
            analysis_result = await llm_service.generate_completion(prompt)
            
            with open(llm_analysis_output, 'w') as f:
                f.write(analysis_result)
            
            result["decompiled_files"].append(llm_analysis_output)
            logger.info(f"Python LLM analysis complete: {llm_analysis_output}")
        
        except Exception as e:
            logger.error(f"Python LLM analysis failed: {str(e)}")
            
        return result
    
    async def _process_c_cpp(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a C/C++ file"""
        logger.info(f"Processing C/C++: {file_path}")
        
        decompiled_dir = os.path.join(result["work_dir"], "decompiled")
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Copy the source for reference
        source_copy = os.path.join(decompiled_dir, os.path.basename(file_path))
        shutil.copy2(file_path, source_copy)
        result["decompiled_files"].append(source_copy)
        
        # Use LLM to analyze the C/C++ code
        llm_analysis_output = os.path.join(decompiled_dir, "cpp_llm_analysis.txt")
        try:
            # Read the C/C++ source
            with open(file_path, 'r') as f:
                cpp_content = f.read()
            
            # Use LLM to analyze the C/C++ code
            prompt = f"""
            As an expert C/C++ developer, analyze this code:
            
            ```cpp
            {cpp_content[:15000]}  # Use first 15k chars to avoid context limit
            ```
            
            Provide:
            1. A summary of what this code does
            2. Key functions and their purposes
            3. Classes, structs, and their methods
            4. Memory management patterns
            5. Dependencies and includes
            6. Overall architecture patterns
            7. Any potential issues or improvements
            
            Format your response as structured text with clear sections.
            """
            
            analysis_result = await llm_service.generate_completion(prompt)
            
            with open(llm_analysis_output, 'w') as f:
                f.write(analysis_result)
            
            result["decompiled_files"].append(llm_analysis_output)
            logger.info(f"C/C++ LLM analysis complete: {llm_analysis_output}")
        
        except Exception as e:
            logger.error(f"C/C++ LLM analysis failed: {str(e)}")
        
        # Try a simple regex-based extraction of functions and structures
        struct_regex = r"(?:struct|class)\s+(\w+)"
        function_regex = r"(?:void|int|bool|char|float|double|auto|unsigned|long|short|\w+\s*\*)\s+(\w+)\s*\([^)]*\)"
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract structures and functions
            structures = re.findall(struct_regex, content)
            functions = re.findall(function_regex, content)
            
            # Save as JSON
            structure_output = os.path.join(decompiled_dir, "cpp_structure.json")
            with open(structure_output, 'w') as f:
                json.dump({
                    "structures": structures,
                    "functions": functions
                }, f, indent=2)
            
            result["decompiled_files"].append(structure_output)
            logger.info(f"C/C++ structure extraction complete: {structure_output}")
            
        except Exception as e:
            logger.error(f"C/C++ structure extraction failed: {str(e)}")
            
        return result
    
    async def _generate_specifications(self, decompiled_files: List[str], work_dir: str) -> List[str]:
        """
        Generate specifications from decompiled files using LLM
        
        Args:
            decompiled_files: List of paths to decompiled files
            work_dir: Working directory
            
        Returns:
            List[str]: Paths to generated specification files
        """
        logger.info(f"Generating specifications from {len(decompiled_files)} decompiled files")
        
        # Create a specifications directory
        spec_dir = os.path.join(work_dir, "specs")
        os.makedirs(spec_dir, exist_ok=True)
        
        spec_files = []
        
        # Process with LLM to extract specifications
        combined_spec_path = os.path.join(spec_dir, "combined_spec.md")
        with open(combined_spec_path, 'w') as spec_file:
            spec_file.write("# Software Specification\n\n")
            spec_file.write("This document contains specifications extracted from the decompiled software.\n\n")
            
            # Process each decompiled file
            for decompiled_file in decompiled_files:
                file_name = os.path.basename(decompiled_file)
                spec_file.write(f"## {file_name}\n\n")
                
                with open(decompiled_file, 'r', errors='replace') as f:
                    try:
                        content = f.read()
                        
                        # Use LLM to generate specification
                        if len(content) > 10000:
                            # Process in chunks
                            chunks = [content[i:i+10000] for i in range(0, len(content), 10000)]
                            chunk_specs = []
                            
                            for i, chunk in enumerate(chunks):
                                prompt = f"""
                                As a software architect, create detailed specifications from this decompiled/analyzed code (chunk {i+1} of {len(chunks)}):
                                
                                ```
                                {chunk}
                                ```
                                
                                Extract:
                                1. Functionality and purpose
                                2. Core components and their relationships
                                3. Data structures
                                4. APIs and interfaces
                                5. Key algorithms
                                
                                Format as Markdown with clear sections.
                                """
                                
                                chunk_spec = await llm_service.generate_completion(prompt)
                                chunk_specs.append(chunk_spec)
                            
                            # Combine chunk specifications
                            file_spec = "\n\n".join(chunk_specs)
                        else:
                            # Process the entire file
                            prompt = f"""
                            As a software architect, create detailed specifications from this decompiled/analyzed code:
                            
                            ```
                            {content}
                            ```
                            
                            Extract:
                            1. Functionality and purpose
                            2. Core components and their relationships
                            3. Data structures
                            4. APIs and interfaces
                            5. Key algorithms
                            
                            Format as Markdown with clear sections.
                            """
                            
                            file_spec = await llm_service.generate_completion(prompt)
                        
                        # Write to specification file
                        spec_file.write(file_spec)
                        spec_file.write("\n\n")
                        
                    except Exception as e:
                        logger.error(f"Error processing {decompiled_file}: {str(e)}")
                        spec_file.write(f"Error processing file: {str(e)}\n\n")
        
        spec_files.append(combined_spec_path)
        logger.info(f"Generated specification: {combined_spec_path}")
        
        # Create individual specification files for each component
        components_prompt = f"""
        Based on all the decompiled code analysis, identify the main software components that should be created.
        List each component with its name, purpose, and required files.
        Format as JSON with fields: name, purpose, files.
        Keep it concise but complete.
        """
        
        try:
            components_response = await llm_service.generate_completion(components_prompt)
            components_spec_path = os.path.join(spec_dir, "components.json")
            
            # Try to parse the response as JSON, if it's not valid JSON, save as text
            try:
                components_json = json.loads(components_response)
                with open(components_spec_path, 'w') as f:
                    json.dump(components_json, f, indent=2)
            except:
                with open(components_spec_path, 'w') as f:
                    f.write(components_response)
            
            spec_files.append(components_spec_path)
            logger.info(f"Generated components specification: {components_spec_path}")
        except Exception as e:
            logger.error(f"Error generating components specification: {str(e)}")
        
        return spec_files
    
    async def _reconstruct_software(self, spec_files: List[str], work_dir: str) -> List[str]:
        """
        Reconstruct software from specifications
        
        Args:
            spec_files: List of paths to specification files
            work_dir: Working directory
            
        Returns:
            List[str]: Paths to reconstructed software files
        """
        logger.info(f"Reconstructing software from {len(spec_files)} specification files")
        
        # Create a reconstruction directory
        reconstructed_dir = os.path.join(work_dir, "reconstructed")
        os.makedirs(reconstructed_dir, exist_ok=True)
        
        reconstructed_files = []
        
        # Read specification files
        specs = []
        for spec_file in spec_files:
            with open(spec_file, 'r') as f:
                specs.append(f.read())
        
        combined_spec = "\n\n".join(specs)
        
        # Use LLM to determine best language and framework for reconstruction
        language_prompt = f"""
        Based on the following specifications:
        
        {combined_spec[:10000]}
        
        Determine the best language and framework to reconstruct this software.
        Consider: Python, JavaScript, Java, C++, or Go.
        Format response as JSON with two fields: language and framework.
        """
        
        language_response = await llm_service.generate_completion(language_prompt)
        
        # Parse the language and framework
        try:
            language_info = json.loads(language_response)
            target_language = language_info.get("language", "python")
            target_framework = language_info.get("framework", "")
        except:
            target_language = "python"
            target_framework = ""
        
        logger.info(f"Reconstructing software in {target_language} with {target_framework if target_framework else 'no specific framework'}")
        
        # Create a project structure
        if target_language.lower() == "python":
            await self._reconstruct_python(reconstructed_dir, reconstructed_files, combined_spec, target_framework)
        elif target_language.lower() in ["javascript", "js"]:
            await self._reconstruct_javascript(reconstructed_dir, reconstructed_files, combined_spec, target_framework)
        elif target_language.lower() in ["java"]:
            await self._reconstruct_java(reconstructed_dir, reconstructed_files, combined_spec, target_framework)
        elif target_language.lower() in ["c++", "cpp"]:
            await self._reconstruct_cpp(reconstructed_dir, reconstructed_files, combined_spec, target_framework)
        elif target_language.lower() in ["go"]:
            await self._reconstruct_go(reconstructed_dir, reconstructed_files, combined_spec, target_framework)
        else:
            # Default to Python if language detection failed
            await self._reconstruct_python(reconstructed_dir, reconstructed_files, combined_spec, "")
        
        return reconstructed_files
    
    async def _reconstruct_python(self, reconstructed_dir: str, reconstructed_files: List[str], spec: str, framework: str) -> None:
        """
        Reconstruct software in Python
        
        Args:
            reconstructed_dir: Directory to store reconstructed files
            reconstructed_files: List to append paths of created files
            spec: Combined specification
            framework: Optional framework to use
        """
        # Create basic Python project structure
        os.makedirs(os.path.join(reconstructed_dir, "src"), exist_ok=True)
        
        # Generate project structure based on specification
        structure_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create a Python project structure with:
        1. Main modules and packages
        2. Directory organization
        3. Key files needed
        
        Format as JSON with a "structure" field containing a list of directories and files.
        Each entry should have "path" and "type" (file or directory).
        """
        
        structure_response = await llm_service.generate_completion(structure_prompt)
        
        # Try to parse the structure response as JSON
        try:
            structure_data = json.loads(structure_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            structure = [
                {"path": "src", "type": "directory"},
                {"path": "src/__init__.py", "type": "file"},
                {"path": "src/main.py", "type": "file"},
                {"path": "src/core", "type": "directory"},
                {"path": "src/core/__init__.py", "type": "file"},
                {"path": "src/utils", "type": "directory"},
                {"path": "src/utils/__init__.py", "type": "file"},
                {"path": "requirements.txt", "type": "file"},
                {"path": "README.md", "type": "file"}
            ]
        
        # Create directories and empty files
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "")
            
            full_path = os.path.join(reconstructed_dir, path)
            
            if item_type == "directory":
                os.makedirs(full_path, exist_ok=True)
            elif item_type == "file":
                # Create an empty file if it doesn't exist
                if not os.path.exists(full_path):
                    with open(full_path, 'w') as f:
                        pass
                reconstructed_files.append(full_path)
        
        # Generate a README
        readme_path = os.path.join(reconstructed_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a README.md with:
        1. Project title and description
        2. Features
        3. Installation instructions
        4. Usage examples
        5. Project structure
        
        Format as Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        if readme_path not in reconstructed_files:
            reconstructed_files.append(readme_path)
        
        # Generate code for each Python file
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "")
            
            if item_type == "file" and path.endswith(".py"):
                file_path = os.path.join(reconstructed_dir, path)
                
                # Generate content for this file
                file_prompt = f"""
                Based on this software specification:
                
                {spec[:8000]}
                
                Generate Python code for the file "{path}".
                Infer the purpose and content from the path and specification.
                Include proper imports, docstrings, and complete implementation.
                
                Format as complete, executable Python code.
                """
                
                file_content = await llm_service.generate_completion(file_prompt)
                
                with open(file_path, 'w') as f:
                    f.write(file_content)
                
                logger.info(f"Generated Python file: {path}")
        
        # Generate requirements.txt
        requirements_path = os.path.join(reconstructed_dir, "requirements.txt")
        requirements_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a requirements.txt file with all necessary Python dependencies.
        Format as a standard requirements.txt with one package per line.
        """
        
        requirements_content = await llm_service.generate_completion(requirements_prompt)
        
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        if requirements_path not in reconstructed_files:
            reconstructed_files.append(requirements_path)
        
        logger.info(f"Reconstructed Python project with {len(reconstructed_files)} files")
    
    async def _reconstruct_javascript(self, reconstructed_dir: str, reconstructed_files: List[str], spec: str, framework: str) -> None:
        """
        Reconstruct software in JavaScript
        
        Args:
            reconstructed_dir: Directory to store reconstructed files
            reconstructed_files: List to append paths of created files
            spec: Combined specification
            framework: Optional framework to use
        """
        # Create basic project structure
        os.makedirs(os.path.join(reconstructed_dir, "src"), exist_ok=True)
        
        # Adjust framework if needed
        if not framework:
            if "react" in spec.lower():
                framework = "react"
            elif "node" in spec.lower() or "express" in spec.lower():
                framework = "express"
            else:
                framework = "vanilla"
        
        # Generate project structure based on specification and framework
        structure_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create a JavaScript project structure using {framework} framework with:
        1. Main modules and components
        2. Directory organization
        3. Key files needed
        
        Format as JSON with a "structure" field containing a list of directories and files.
        Each entry should have "path" and "type" (file or directory).
        """
        
        structure_response = await llm_service.generate_completion(structure_prompt)
        
        # Try to parse the structure response as JSON
        try:
            structure_data = json.loads(structure_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure based on framework
            if framework == "react":
                structure = [
                    {"path": "public", "type": "directory"},
                    {"path": "public/index.html", "type": "file"},
                    {"path": "src", "type": "directory"},
                    {"path": "src/App.js", "type": "file"},
                    {"path": "src/index.js", "type": "file"},
                    {"path": "src/components", "type": "directory"},
                    {"path": "src/styles", "type": "directory"},
                    {"path": "package.json", "type": "file"},
                    {"path": "README.md", "type": "file"}
                ]
            elif framework == "express":
                structure = [
                    {"path": "src", "type": "directory"},
                    {"path": "src/index.js", "type": "file"},
                    {"path": "src/routes", "type": "directory"},
                    {"path": "src/controllers", "type": "directory"},
                    {"path": "src/models", "type": "directory"},
                    {"path": "src/middlewares", "type": "directory"},
                    {"path": "src/utils", "type": "directory"},
                    {"path": "package.json", "type": "file"},
                    {"path": "README.md", "type": "file"}
                ]
            else:
                structure = [
                    {"path": "src", "type": "directory"},
                    {"path": "src/index.js", "type": "file"},
                    {"path": "src/modules", "type": "directory"},
                    {"path": "src/utils", "type": "directory"},
                    {"path": "package.json", "type": "file"},
                    {"path": "README.md", "type": "file"}
                ]
        
        # Create directories and empty files
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "")
            
            full_path = os.path.join(reconstructed_dir, path)
            
            if item_type == "directory":
                os.makedirs(full_path, exist_ok=True)
            elif item_type == "file":
                # Create an empty file if it doesn't exist
                if not os.path.exists(full_path):
                    with open(full_path, 'w') as f:
                        pass
                reconstructed_files.append(full_path)
        
        # Generate a README
        readme_path = os.path.join(reconstructed_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a README.md for a JavaScript project using {framework} framework with:
        1. Project title and description
        2. Features
        3. Installation instructions
        4. Usage examples
        5. Project structure
        
        Format as Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        if readme_path not in reconstructed_files:
            reconstructed_files.append(readme_path)
        
        # Generate package.json
        package_path = os.path.join(reconstructed_dir, "package.json")
        package_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a package.json file for a JavaScript project using {framework} framework.
        Include all necessary dependencies, scripts, and metadata.
        Format as valid JSON.
        """
        
        package_content = await llm_service.generate_completion(package_prompt)
        
        # Try to parse and format the package.json content
        try:
            package_data = json.loads(package_content)
            with open(package_path, 'w') as f:
                json.dump(package_data, f, indent=2)
        except:
            # If JSON parsing fails, just write the content as is
            with open(package_path, 'w') as f:
                f.write(package_content)
        
        if package_path not in reconstructed_files:
            reconstructed_files.append(package_path)
        
        # Generate code for each JavaScript file
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "")
            
            if item_type == "file" and (path.endswith(".js") or path.endswith(".jsx")):
                file_path = os.path.join(reconstructed_dir, path)
                
                # Generate content for this file
                file_prompt = f"""
                Based on this software specification:
                
                {spec[:8000]}
                
                Generate JavaScript code for the file "{path}" in a {framework} project.
                Infer the purpose and content from the path and specification.
                Include proper imports and complete implementation.
                
                Format as complete, executable JavaScript code.
                """
                
                file_content = await llm_service.generate_completion(file_prompt)
                
                with open(file_path, 'w') as f:
                    f.write(file_content)
                
                logger.info(f"Generated JavaScript file: {path}")
        
        # For React, generate HTML template if needed
        if framework == "react" and "public/index.html" in [item.get("path") for item in structure if item.get("type") == "file"]:
            html_path = os.path.join(reconstructed_dir, "public/index.html")
            html_prompt = """
            Generate a basic index.html template for a React application.
            Include root div, proper meta tags, and basic structure.
            """
            
            html_content = await llm_service.generate_completion(html_prompt)
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info("Generated React HTML template")
        
        logger.info(f"Reconstructed JavaScript project with {len(reconstructed_files)} files")
    
    async def _reconstruct_java(self, reconstructed_dir: str, reconstructed_files: List[str], spec: str, framework: str) -> None:
        """Reconstruct software in Java"""
        # Create basic Java project structure
        src_main_java = os.path.join(reconstructed_dir, "src", "main", "java")
        src_test_java = os.path.join(reconstructed_dir, "src", "test", "java")
        os.makedirs(src_main_java, exist_ok=True)
        os.makedirs(src_test_java, exist_ok=True)
        
        # Determine package structure from spec
        package_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Determine the main Java package name following standard naming conventions.
        Format as a single line, e.g., "com.example.project".
        """
        
        package_name = await llm_service.generate_completion(package_prompt)
        package_name = package_name.strip().replace("\n", "")
        
        if not package_name or " " in package_name:
            package_name = "com.example.project"
        
        # Create package directory structure
        package_path = os.path.join(src_main_java, *package_name.split("."))
        os.makedirs(package_path, exist_ok=True)
        
        # Generate project structure based on specification
        structure_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create a Java project structure with package {package_name} including:
        1. Main classes and their relationships
        2. Directory organization
        3. Key files needed
        
        Format as JSON with a "structure" field containing a list of Java files.
        Each entry should have "path" relative to src/main/java and "type" (class, interface, enum).
        """
        
        structure_response = await llm_service.generate_completion(structure_prompt)
        
        # Try to parse the structure response as JSON
        try:
            structure_data = json.loads(structure_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            package_parts = package_name.split(".")
            structure = [
                {"path": f"{'/'.join(package_parts)}/Main.java", "type": "class"},
                {"path": f"{'/'.join(package_parts)}/model/Model.java", "type": "class"},
                {"path": f"{'/'.join(package_parts)}/service/Service.java", "type": "interface"},
                {"path": f"{'/'.join(package_parts)}/service/impl/ServiceImpl.java", "type": "class"},
                {"path": f"{'/'.join(package_parts)}/util/Utility.java", "type": "class"}
            ]
        
        # Create directories and empty files
        for item in structure:
            path = item.get("path", "")
            
            full_path = os.path.join(src_main_java, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Create an empty file if it doesn't exist
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass
            reconstructed_files.append(full_path)
        
        # Generate Maven pom.xml
        pom_path = os.path.join(reconstructed_dir, "pom.xml")
        pom_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a Maven pom.xml file for a Java project with:
        1. Group ID: {package_name.split(".")[0]}.{package_name.split(".")[1]}
        2. Artifact ID: {package_name.split(".")[-1]}
        3. All necessary dependencies
        4. Build plugins
        5. Project properties
        
        Format as valid XML.
        """
        
        pom_content = await llm_service.generate_completion(pom_prompt)
        
        with open(pom_path, 'w') as f:
            f.write(pom_content)
        
        reconstructed_files.append(pom_path)
        
        # Generate a README
        readme_path = os.path.join(reconstructed_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a README.md for a Java project with:
        1. Project title and description
        2. Features
        3. Installation instructions
        4. Usage examples
        5. Project structure
        
        Format as Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        reconstructed_files.append(readme_path)
        
        # Generate code for each Java file
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "class")
            
            file_path = os.path.join(src_main_java, path)
            
            # Generate content for this file
            file_prompt = f"""
            Based on this software specification:
            
            {spec[:8000]}
            
            Generate Java code for the file "{path}" of type {item_type}.
            Infer the purpose and content from the path and specification.
            Include proper package declaration, imports, and complete implementation.
            Use package {package_name} with appropriate subpackages based on the file path.
            
            Format as complete, executable Java code.
            """
            
            file_content = await llm_service.generate_completion(file_prompt)
            
            with open(file_path, 'w') as f:
                f.write(file_content)
            
            logger.info(f"Generated Java file: {path}")
        
        logger.info(f"Reconstructed Java project with {len(reconstructed_files)} files")
    
    async def _reconstruct_cpp(self, reconstructed_dir: str, reconstructed_files: List[str], spec: str, framework: str) -> None:
        """Reconstruct software in C++"""
        # Create basic C++ project structure
        os.makedirs(os.path.join(reconstructed_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(reconstructed_dir, "include"), exist_ok=True)
        os.makedirs(os.path.join(reconstructed_dir, "build"), exist_ok=True)
        
        # Generate project structure based on specification
        structure_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create a C++ project structure with:
        1. Main classes and their relationships
        2. Header and implementation files
        3. Directory organization
        
        Format as JSON with a "structure" field containing a list of C++ files.
        Each entry should have "path" and "type" (header or implementation).
        Headers should be in include/ and implementations in src/.
        """
        
        structure_response = await llm_service.generate_completion(structure_prompt)
        
        # Try to parse the structure response as JSON
        try:
            structure_data = json.loads(structure_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            structure = [
                {"path": "include/Main.h", "type": "header"},
                {"path": "src/Main.cpp", "type": "implementation"},
                {"path": "include/Core.h", "type": "header"},
                {"path": "src/Core.cpp", "type": "implementation"},
                {"path": "include/Utils.h", "type": "header"},
                {"path": "src/Utils.cpp", "type": "implementation"},
                {"path": "src/main.cpp", "type": "implementation"}
            ]
        
        # Create directories and empty files
        for item in structure:
            path = item.get("path", "")
            
            full_path = os.path.join(reconstructed_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Create an empty file if it doesn't exist
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass
            reconstructed_files.append(full_path)
        
        # Generate CMakeLists.txt
        cmake_path = os.path.join(reconstructed_dir, "CMakeLists.txt")
        cmake_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a CMakeLists.txt file for a C++ project with:
        1. Project name derived from the specification
        2. C++ standard (C++17 or later)
        3. All source files properly included
        4. Build targets
        5. Any required libraries
        
        Format as a valid CMake file.
        """
        
        cmake_content = await llm_service.generate_completion(cmake_prompt)
        
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        reconstructed_files.append(cmake_path)
        
        # Generate a README
        readme_path = os.path.join(reconstructed_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a README.md for a C++ project with:
        1. Project title and description
        2. Features
        3. Build instructions using CMake
        4. Usage examples
        5. Project structure
        
        Format as Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        reconstructed_files.append(readme_path)
        
        # Generate code for each C++ file
        for item in structure:
            path = item.get("path", "")
            item_type = item.get("type", "")
            
            file_path = os.path.join(reconstructed_dir, path)
            
            # Generate content for this file
            file_prompt = f"""
            Based on this software specification:
            
            {spec[:8000]}
            
            Generate C++ code for the file "{path}" of type {item_type}.
            Infer the purpose and content from the path and specification.
            Include proper includes, namespaces, and complete implementation.
            
            Format as complete, executable C++ code.
            """
            
            file_content = await llm_service.generate_completion(file_prompt)
            
            with open(file_path, 'w') as f:
                f.write(file_content)
            
            logger.info(f"Generated C++ file: {path}")
        
        logger.info(f"Reconstructed C++ project with {len(reconstructed_files)} files")
    
    async def _reconstruct_go(self, reconstructed_dir: str, reconstructed_files: List[str], spec: str, framework: str) -> None:
        """Reconstruct software in Go"""
        # Create basic Go project structure
        os.makedirs(os.path.join(reconstructed_dir, "cmd"), exist_ok=True)
        os.makedirs(os.path.join(reconstructed_dir, "internal"), exist_ok=True)
        os.makedirs(os.path.join(reconstructed_dir, "pkg"), exist_ok=True)
        
        # Determine the module name
        module_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Determine a suitable Go module name following standard naming conventions.
        Format as a single line, e.g., "github.com/example/project".
        """
        
        module_name = await llm_service.generate_completion(module_prompt)
        module_name = module_name.strip().replace("\n", "")
        
        if not module_name or " " in module_name:
            module_name = "github.com/example/project"
        
        # Generate project structure based on specification
        structure_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create a Go project structure with module name {module_name} following:
        1. Standard Go project layout
        2. Main packages and their relationships
        3. Directory organization
        
        Format as JSON with a "structure" field containing a list of Go files.
        Each entry should have "path" and "package" (package name).
        """
        
        structure_response = await llm_service.generate_completion(structure_prompt)
        
        # Try to parse the structure response as JSON
        try:
            structure_data = json.loads(structure_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            structure = [
                {"path": "cmd/main.go", "package": "main"},
                {"path": "internal/app/app.go", "package": "app"},
                {"path": "internal/config/config.go", "package": "config"},
                {"path": "pkg/util/util.go", "package": "util"}
            ]
        
        # Create go.mod file
        go_mod_path = os.path.join(reconstructed_dir, "go.mod")
        with open(go_mod_path, 'w') as f:
            f.write(f"module {module_name}\n\ngo 1.19\n")
        
        reconstructed_files.append(go_mod_path)
        
        # Create directories and empty files
        for item in structure:
            path = item.get("path", "")
            
            full_path = os.path.join(reconstructed_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Create an empty file if it doesn't exist
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass
            reconstructed_files.append(full_path)
        
        # Generate a README
        readme_path = os.path.join(reconstructed_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a README.md for a Go project with:
        1. Project title and description
        2. Features
        3. Installation instructions
        4. Usage examples
        5. Project structure
        
        Format as Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        reconstructed_files.append(readme_path)
        
        # Generate code for each Go file
        for item in structure:
            path = item.get("path", "")
            package = item.get("package", "main")
            
            file_path = os.path.join(reconstructed_dir, path)
            
            # Generate content for this file
            file_prompt = f"""
            Based on this software specification:
            
            {spec[:8000]}
            
            Generate Go code for the file "{path}" in package "{package}".
            Infer the purpose and content from the path and specification.
            Include proper imports and complete implementation.
            Use module name {module_name} for imports when needed.
            
            Format as complete, executable Go code.
            """
            
            file_content = await llm_service.generate_completion(file_prompt)
            
            with open(file_path, 'w') as f:
                f.write(file_content)
            
            logger.info(f"Generated Go file: {path}")
        
        # Generate go.sum (empty)
        go_sum_path = os.path.join(reconstructed_dir, "go.sum")
        with open(go_sum_path, 'w') as f:
            pass
        
        reconstructed_files.append(go_sum_path)
        
        logger.info(f"Reconstructed Go project with {len(reconstructed_files)} files")
    
    async def mimic_software(self, spec_files: List[str], target_language: str, software_id: int) -> Dict[str, Any]:
        """
        Generate mimicked software in a target language
        
        Args:
            spec_files: List of paths to specification files
            target_language: Target language for the new software
            software_id: ID of the original software
            
        Returns:
            Dict[str, Any]: Result of the mimicry process
        """
        logger.info(f"Mimicking software in {target_language} based on {len(spec_files)} specification files")
        
        # Create a unique working directory for the mimicry
        work_subdir = f"mimic_{software_id}_{uuid.uuid4().hex[:8]}"
        mimic_work_dir = os.path.join(self.work_dir, work_subdir)
        mimicked_dir = os.path.join(mimic_work_dir, f"mimicked_{target_language}")
        os.makedirs(mimicked_dir, exist_ok=True)
        
        # Load specifications
        specs = []
        for spec_file in spec_files:
            with open(spec_file, 'r') as f:
                specs.append(f.read())
        
        combined_spec = "\n\n".join(specs)
        
        # Set up result
        result = {
            "software_id": software_id,
            "target_language": target_language,
            "mimicked_dir": mimicked_dir,
            "mimicked_files": [],
            "status": ProcessStatus.PENDING
        }
        
        try:
            # Generate mimicked code based on the target language
            if target_language.lower() in ["python", "py"]:
                await self._mimic_as_python(mimicked_dir, result["mimicked_files"], combined_spec)
            elif target_language.lower() in ["javascript", "js"]:
                await self._mimic_as_javascript(mimicked_dir, result["mimicked_files"], combined_spec)
            elif target_language.lower() in ["c++", "cpp"]:
                await self._mimic_as_cpp(mimicked_dir, result["mimicked_files"], combined_spec)
            elif target_language.lower() in ["java"]:
                await self._mimic_as_java(mimicked_dir, result["mimicked_files"], combined_spec)
            elif target_language.lower() in ["go", "golang"]:
                await self._mimic_as_go(mimicked_dir, result["mimicked_files"], combined_spec)
            else:
                # Default to Python if the target language is not supported
                await self._mimic_as_python(mimicked_dir, result["mimicked_files"], combined_spec)
            
            result["status"] = ProcessStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error during mimicry: {str(e)}", exc_info=True)
            result["status"] = ProcessStatus.ERROR
            result["error_message"] = str(e)
        
        return result
    
    async def _mimic_as_python(self, mimicked_dir: str, mimicked_files: List[str], spec: str) -> None:
        """
        Generate a Python implementation that mimics the original software
        with enhanced capabilities
        
        Args:
            mimicked_dir: Directory to store the mimicked software
            mimicked_files: List to append generated file paths to
            spec: Combined specification of the original software
        """
        # Create more sophisticated Python structure
        os.makedirs(os.path.join(mimicked_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "core"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "utils"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "data"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "tests"), exist_ok=True)
        
        # Create improved version with LLM
        enhancement_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create an enhanced Python implementation with:
        1. Modern architecture (clean architecture or similar)
        2. Improved error handling
        3. Comprehensive logging
        4. Enhanced security
        5. Better performance
        6. Type hints
        
        First, outline the project structure as JSON with a "structure" field containing:
        - List of files with "path" and "purpose" fields
        - Group components logically
        """
        
        enhancement_response = await llm_service.generate_completion(enhancement_prompt)
        
        # Try to parse the enhancement response as JSON
        try:
            structure_data = json.loads(enhancement_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            structure = [
                {"path": "src/__init__.py", "purpose": "Package initialization"},
                {"path": "src/main.py", "purpose": "Application entry point"},
                {"path": "src/core/__init__.py", "purpose": "Core package initialization"},
                {"path": "src/core/engine.py", "purpose": "Main engine implementation"},
                {"path": "src/utils/__init__.py", "purpose": "Utils package initialization"},
                {"path": "src/utils/helpers.py", "purpose": "Helper functions"},
                {"path": "src/utils/logging.py", "purpose": "Logging configuration"},
                {"path": "src/data/__init__.py", "purpose": "Data package initialization"},
                {"path": "src/data/processor.py", "purpose": "Data processing implementation"},
                {"path": "tests/__init__.py", "purpose": "Tests package initialization"},
                {"path": "tests/test_core.py", "purpose": "Core tests"},
                {"path": "requirements.txt", "purpose": "Project dependencies"},
                {"path": "setup.py", "purpose": "Package setup file"},
                {"path": "README.md", "purpose": "Project documentation"}
            ]
        
        # Create __init__.py files in each directory if needed
        for directory in ["src", "src/core", "src/utils", "src/data", "tests"]:
            init_path = os.path.join(mimicked_dir, directory, "__init__.py")
            if not os.path.exists(init_path) and not any(item.get("path") == f"{directory}/__init__.py" for item in structure):
                structure.append({"path": f"{directory}/__init__.py", "purpose": "Package initialization"})
        
        # Create files
        for item in structure:
            path = item.get("path", "")
            purpose = item.get("purpose", "")
            
            full_path = os.path.join(mimicked_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Generate content for this file
            if path.endswith(".py"):
                file_prompt = f"""
                Based on this software specification:
                
                {spec[:8000]}
                
                Generate Python code for the file "{path}" with purpose: "{purpose}".
                This should be an enhanced implementation with:
                1. Modern Python 3.8+ features
                2. Type hints
                3. Comprehensive error handling
                4. Proper logging
                5. Clean architecture
                6. Documentation
                
                Format as complete, professional-grade Python code.
                """
                
                file_content = await llm_service.generate_completion(file_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(file_content)
                
                logger.info(f"Generated Python file: {path}")
                mimicked_files.append(full_path)
            elif path == "requirements.txt":
                requirements_prompt = f"""
                Based on this software specification:
                
                {spec[:5000]}
                
                Generate a comprehensive requirements.txt with all dependencies needed for an enhanced version.
                Include modern libraries for performance, security, and functionality.
                """
                
                requirements_content = await llm_service.generate_completion(requirements_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(requirements_content)
                
                logger.info("Generated requirements.txt")
                mimicked_files.append(full_path)
            elif path == "setup.py":
                setup_prompt = f"""
                Based on this software specification:
                
                {spec[:5000]}
                
                Generate a setup.py file for packaging this Python project.
                Include all necessary metadata, dependencies, and entry points.
                """
                
                setup_content = await llm_service.generate_completion(setup_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(setup_content)
                
                logger.info("Generated setup.py")
                mimicked_files.append(full_path)
            elif path == "README.md":
                readme_prompt = f"""
                Based on this software specification:
                
                {spec[:5000]}
                
                Create a comprehensive README.md with:
                1. Project title and description
                2. Enhanced features compared to the original
                3. Installation instructions
                4. Usage examples with code snippets
                5. API documentation
                6. License (MIT)
                
                Format as professional Markdown.
                """
                
                readme_content = await llm_service.generate_completion(readme_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(readme_content)
                
                logger.info("Generated README.md")
                mimicked_files.append(full_path)
        
        logger.info(f"Mimicked Python project with {len(mimicked_files)} files")
    
    async def _mimic_as_javascript(self, mimicked_dir: str, mimicked_files: List[str], spec: str) -> None:
        """
        Generate a JavaScript implementation that mimics the original software
        
        Args:
            mimicked_dir: Directory to store the mimicked software
            mimicked_files: List to append generated file paths to
            spec: Combined specification of the original software
        """
        # Create node.js structure
        os.makedirs(os.path.join(mimicked_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "core"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "utils"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "src", "data"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "tests"), exist_ok=True)
        
        # Determine if we should use a framework based on the spec
        framework_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Recommend the best JavaScript framework to use for an enhanced implementation.
        Options: express, react, vue, angular, or none/vanilla.
        Give a single-word answer.
        """
        
        framework_response = await llm_service.generate_completion(framework_prompt)
        framework = framework_response.strip().lower()
        
        if framework not in ["express", "react", "vue", "angular"]:
            framework = "express"  # Default to Express for backend
        
        # Create improved version with LLM
        enhancement_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create an enhanced JavaScript implementation using {framework} framework with:
        1. Modern architecture
        2. Improved error handling
        3. Comprehensive logging
        4. Enhanced security
        5. Better performance
        6. ES6+ features
        
        First, outline the project structure as JSON with a "structure" field containing:
        - List of files with "path" and "purpose" fields
        - Group components logically
        """
        
        enhancement_response = await llm_service.generate_completion(enhancement_prompt)
        
        # Try to parse the enhancement response as JSON
        try:
            structure_data = json.loads(enhancement_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure based on the framework
            if framework == "express":
                structure = [
                    {"path": "src/index.js", "purpose": "Application entry point"},
                    {"path": "src/app.js", "purpose": "Express application setup"},
                    {"path": "src/core/engine.js", "purpose": "Main engine implementation"},
                    {"path": "src/utils/logger.js", "purpose": "Logging utility"},
                    {"path": "src/utils/config.js", "purpose": "Configuration manager"},
                    {"path": "src/data/processor.js", "purpose": "Data processing implementation"},
                    {"path": "src/routes/index.js", "purpose": "Route definitions"},
                    {"path": "src/controllers/index.js", "purpose": "Request controllers"},
                    {"path": "src/models/index.js", "purpose": "Data models"},
                    {"path": "tests/core.test.js", "purpose": "Core tests"},
                    {"path": "package.json", "purpose": "Project configuration"},
                    {"path": "README.md", "purpose": "Project documentation"}
                ]
            elif framework == "react":
                structure = [
                    {"path": "src/index.js", "purpose": "React entry point"},
                    {"path": "src/App.js", "purpose": "Main React component"},
                    {"path": "src/components/Header.js", "purpose": "Header component"},
                    {"path": "src/components/Footer.js", "purpose": "Footer component"},
                    {"path": "src/core/engine.js", "purpose": "Application logic"},
                    {"path": "src/utils/logger.js", "purpose": "Logging utility"},
                    {"path": "src/utils/api.js", "purpose": "API client"},
                    {"path": "src/data/store.js", "purpose": "State management"},
                    {"path": "public/index.html", "purpose": "HTML template"},
                    {"path": "tests/App.test.js", "purpose": "Application tests"},
                    {"path": "package.json", "purpose": "Project configuration"},
                    {"path": "README.md", "purpose": "Project documentation"}
                ]
            else:
                structure = [
                    {"path": "src/index.js", "purpose": "Application entry point"},
                    {"path": "src/core/engine.js", "purpose": "Main engine implementation"},
                    {"path": "src/utils/logger.js", "purpose": "Logging utility"},
                    {"path": "src/utils/helpers.js", "purpose": "Helper functions"},
                    {"path": "src/data/processor.js", "purpose": "Data processing implementation"},
                    {"path": "tests/core.test.js", "purpose": "Core tests"},
                    {"path": "package.json", "purpose": "Project configuration"},
                    {"path": "README.md", "purpose": "Project documentation"}
                ]
        
        # Create directories for all files
        for item in structure:
            path = item.get("path", "")
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(os.path.join(mimicked_dir, directory), exist_ok=True)
        
        # Generate files
        for item in structure:
            path = item.get("path", "")
            purpose = item.get("purpose", "")
            
            full_path = os.path.join(mimicked_dir, path)
            
            if path.endswith(".js") or path.endswith(".jsx"):
                file_prompt = f"""
                Based on this software specification:
                
                {spec[:8000]}
                
                Generate JavaScript code for the file "{path}" with purpose: "{purpose}".
                This should be an enhanced implementation using {framework} framework with:
                1. Modern ES6+ features
                2. Robust error handling
                3. Proper logging
                4. Clean architecture
                5. Documentation (JSDoc)
                
                Format as complete, professional-grade JavaScript code.
                """
                
                file_content = await llm_service.generate_completion(file_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(file_content)
                
                logger.info(f"Generated JavaScript file: {path}")
                mimicked_files.append(full_path)
            elif path == "package.json":
                package_prompt = f"""
                Based on this software specification:
                
                {spec[:5000]}
                
                Generate a package.json file for a {framework} project with:
                1. All necessary dependencies
                2. Dev dependencies (testing, linting, etc.)
                3. Scripts (start, test, build, etc.)
                4. Metadata (name, version, description, etc.)
                
                Format as valid JSON.
                """
                
                package_content = await llm_service.generate_completion(package_prompt)
                
                try:
                    # Ensure it's valid JSON
                    package_data = json.loads(package_content)
                    with open(full_path, 'w') as f:
                        json.dump(package_data, f, indent=2)
                except:
                    # If not valid JSON, just write as is
                    with open(full_path, 'w') as f:
                        f.write(package_content)
                
                logger.info("Generated package.json")
                mimicked_files.append(full_path)
            elif path == "README.md":
                readme_prompt = f"""
                Based on this software specification:
                
                {spec[:5000]}
                
                Create a comprehensive README.md for a {framework} project with:
                1. Project title and description
                2. Enhanced features compared to the original
                3. Installation instructions
                4. Usage examples with code snippets
                5. API documentation
                6. License (MIT)
                
                Format as professional Markdown.
                """
                
                readme_content = await llm_service.generate_completion(readme_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(readme_content)
                
                logger.info("Generated README.md")
                mimicked_files.append(full_path)
            elif path.endswith(".html"):
                html_prompt = f"""
                Generate a clean HTML template for a {framework} application based on:
                
                {spec[:3000]}
                
                Include all necessary elements, meta tags, and structure.
                Format as valid HTML5.
                """
                
                html_content = await llm_service.generate_completion(html_prompt)
                
                with open(full_path, 'w') as f:
                    f.write(html_content)
                
                logger.info(f"Generated HTML file: {path}")
                mimicked_files.append(full_path)
        
        # Add special files based on framework
        if framework == "react":
            # Create .gitignore
            gitignore_path = os.path.join(mimicked_dir, ".gitignore")
            with open(gitignore_path, 'w') as f:
                f.write("node_modules\n.env\nbuild\n.DS_Store\n.vscode\n")
            mimicked_files.append(gitignore_path)
            
            # Create .env
            env_path = os.path.join(mimicked_dir, ".env")
            with open(env_path, 'w') as f:
                f.write("REACT_APP_API_URL=http://localhost:5000\n")
            mimicked_files.append(env_path)
        elif framework == "express":
            # Create .gitignore
            gitignore_path = os.path.join(mimicked_dir, ".gitignore")
            with open(gitignore_path, 'w') as f:
                f.write("node_modules\n.env\nlogs\n.DS_Store\n.vscode\n")
            mimicked_files.append(gitignore_path)
            
            # Create .env
            env_path = os.path.join(mimicked_dir, ".env")
            with open(env_path, 'w') as f:
                f.write("PORT=5000\nNODE_ENV=development\n")
            mimicked_files.append(env_path)
        
        logger.info(f"Mimicked JavaScript project with {len(mimicked_files)} files using {framework} framework")
    
    async def _mimic_as_cpp(self, mimicked_dir: str, mimicked_files: List[str], spec: str) -> None:
        """
        Generate a C++ implementation that mimics the original software
        
        Args:
            mimicked_dir: Directory to store the mimicked software
            mimicked_files: List to append generated file paths to
            spec: Combined specification of the original software
        """
        # Create directories for C++ project
        os.makedirs(os.path.join(mimicked_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "include"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "build"), exist_ok=True)
        os.makedirs(os.path.join(mimicked_dir, "test"), exist_ok=True)
        
        # Create improved version with LLM
        enhancement_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create an enhanced C++ implementation with:
        1. Modern C++17/20 features
        2. Improved error handling
        3. Memory safety
        4. Performance optimizations
        5. Clean architecture
        
        First, outline the project structure as JSON with a "structure" field containing:
        - List of files with "path" and "purpose" fields
        - Include both header (.h) and implementation (.cpp) files
        - Group components logically
        """
        
        enhancement_response = await llm_service.generate_completion(enhancement_prompt)
        
        # Try to parse the enhancement response as JSON
        try:
            structure_data = json.loads(enhancement_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            structure = [
                {"path": "include/Application.h", "purpose": "Main application header"},
                {"path": "src/Application.cpp", "purpose": "Main application implementation"},
                {"path": "include/Core.h", "purpose": "Core functionality header"},
                {"path": "src/Core.cpp", "purpose": "Core functionality implementation"},
                {"path": "include/Utils.h", "purpose": "Utilities header"},
                {"path": "src/Utils.cpp", "purpose": "Utilities implementation"},
                {"path": "include/DataProcessor.h", "purpose": "Data processing header"},
                {"path": "src/DataProcessor.cpp", "purpose": "Data processing implementation"},
                {"path": "src/main.cpp", "purpose": "Entry point"},
                {"path": "test/CoreTest.cpp", "purpose": "Core tests"},
                {"path": "CMakeLists.txt", "purpose": "CMake build configuration"},
                {"path": "README.md", "purpose": "Project documentation"}
            ]
        
        # Generate CMakeLists.txt
        cmake_path = os.path.join(mimicked_dir, "CMakeLists.txt")
        cmake_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a CMakeLists.txt file for a modern C++ project with:
        1. C++17 or C++20 standard
        2. All source files properly included
        3. Test configuration
        4. Installation instructions
        5. Necessary dependencies
        
        Format as a complete, professional CMake configuration.
        """
        
        cmake_content = await llm_service.generate_completion(cmake_prompt)
        
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        logger.info("Generated CMakeLists.txt")
        mimicked_files.append(cmake_path)
        
        # Generate README
        readme_path = os.path.join(mimicked_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a comprehensive README.md for a C++ project with:
        1. Project title and description
        2. Enhanced features compared to the original
        3. Build instructions using CMake
        4. Usage examples
        5. API documentation
        6. License (MIT)
        
        Format as professional Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("Generated README.md")
        mimicked_files.append(readme_path)
        
        # Generate all C++ files
        for item in structure:
            path = item.get("path", "")
            purpose = item.get("purpose", "")
            
            if not path.endswith(('.h', '.hpp', '.cpp', '.cc')):
                continue
                
            full_path = os.path.join(mimicked_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            file_type = "header" if path.endswith(('.h', '.hpp')) else "implementation"
            
            file_prompt = f"""
            Based on this software specification:
            
            {spec[:8000]}
            
            Generate C++ code for the {file_type} file "{path}" with purpose: "{purpose}".
            This should be an enhanced implementation with:
            1. Modern C++17/20 features
            2. Robust error handling with exceptions
            3. Memory safety (smart pointers, RAII)
            4. Clean architecture
            5. Documentation
            
            Format as complete, professional-grade C++ code.
            """
            
            file_content = await llm_service.generate_completion(file_prompt)
            
            with open(full_path, 'w') as f:
                f.write(file_content)
            
            logger.info(f"Generated C++ file: {path}")
            mimicked_files.append(full_path)
        
        # Add .gitignore
        gitignore_path = os.path.join(mimicked_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write("build/\n.vscode/\n.idea/\nCMakeFiles/\nCMakeCache.txt\n*.o\n*.exe\n*.out\n")
        mimicked_files.append(gitignore_path)
        
        logger.info(f"Mimicked C++ project with {len(mimicked_files)} files")
    
    async def _mimic_as_java(self, mimicked_dir: str, mimicked_files: List[str], spec: str) -> None:
        """
        Generate a Java implementation that mimics the original software
        
        Args:
            mimicked_dir: Directory to store the mimicked software
            mimicked_files: List to append generated file paths to
            spec: Combined specification of the original software
        """
        # Create Maven project structure
        src_main_java = os.path.join(mimicked_dir, "src", "main", "java")
        src_test_java = os.path.join(mimicked_dir, "src", "test", "java")
        src_main_resources = os.path.join(mimicked_dir, "src", "main", "resources")
        
        os.makedirs(src_main_java, exist_ok=True)
        os.makedirs(src_test_java, exist_ok=True)
        os.makedirs(src_main_resources, exist_ok=True)
        
        # Determine package structure
        package_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Suggest a proper Java package name following standard conventions (e.g., com.company.project).
        Provide a single line response.
        """
        
        package_name = await llm_service.generate_completion(package_prompt)
        package_name = package_name.strip().replace("\n", "")
        
        if not package_name or " " in package_name:
            package_name = "com.enhanced.application"
        
        # Create package directories
        package_path = os.path.join(src_main_java, *package_name.split("."))
        os.makedirs(package_path, exist_ok=True)
        
        # Create improved version with LLM
        enhancement_prompt = f"""
        Based on this software specification:
        
        {spec[:10000]}
        
        Create an enhanced Java implementation with:
        1. Modern Java features (Java 11+)
        2. Improved error handling
        3. Clean architecture (MVC, MVVM, Clean, etc.)
        4. Performance optimizations
        5. Dependency injection
        
        First, outline the project structure as JSON with a "structure" field containing:
        - List of Java files with "path" (relative to src/main/java and package {package_name}) and "purpose" fields
        - Group components logically (model, view, controller, service, etc.)
        """
        
        enhancement_response = await llm_service.generate_completion(enhancement_prompt)
        
        # Try to parse the enhancement response as JSON
        try:
            structure_data = json.loads(enhancement_response)
            structure = structure_data.get("structure", [])
        except:
            # If JSON parsing fails, create a basic structure
            package_parts = package_name.split(".")
            base_path = "/".join(package_parts)
            structure = [
                {"path": f"{base_path}/Application.java", "purpose": "Main application class"},
                {"path": f"{base_path}/Main.java", "purpose": "Entry point"},
                {"path": f"{base_path}/model/Model.java", "purpose": "Data model"},
                {"path": f"{base_path}/service/Service.java", "purpose": "Service interface"},
                {"path": f"{base_path}/service/impl/ServiceImpl.java", "purpose": "Service implementation"},
                {"path": f"{base_path}/controller/Controller.java", "purpose": "Controller class"},
                {"path": f"{base_path}/util/Utility.java", "purpose": "Utility functions"},
                {"path": f"{base_path}/config/Configuration.java", "purpose": "Application configuration"}
            ]
        
        # Generate pom.xml
        pom_path = os.path.join(mimicked_dir, "pom.xml")
        pom_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate a Maven pom.xml file for a modern Java project with:
        1. Java 11+ configuration
        2. All necessary dependencies
        3. Build plugins
        4. Test framework (JUnit 5)
        5. Group ID: {package_name.split('.')[0]}.{package_name.split('.')[1]}
        6. Artifact ID: {package_name.split('.')[-1]}
        
        Format as complete, professional Maven configuration.
        """
        
        pom_content = await llm_service.generate_completion(pom_prompt)
        
        with open(pom_path, 'w') as f:
            f.write(pom_content)
        
        logger.info("Generated pom.xml")
        mimicked_files.append(pom_path)
        
        # Generate README
        readme_path = os.path.join(mimicked_dir, "README.md")
        readme_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Create a comprehensive README.md for a Java project with:
        1. Project title and description
        2. Enhanced features compared to the original
        3. Build instructions using Maven
        4. Usage examples
        5. API documentation
        6. License (MIT)
        
        Format as professional Markdown.
        """
        
        readme_content = await llm_service.generate_completion(readme_prompt)
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info("Generated README.md")
        mimicked_files.append(readme_path)
        
        # Generate application.properties
        properties_path = os.path.join(src_main_resources, "application.properties")
        properties_prompt = f"""
        Based on this software specification:
        
        {spec[:5000]}
        
        Generate an application.properties file for a Java application.
        Include configuration for logging, database (if needed), and application-specific settings.
        """
        
        properties_content = await llm_service.generate_completion(properties_prompt)
        
        with open(properties_path, 'w') as f:
            f.write(properties_content)
        
        logger.info("Generated application.properties")
        mimicked_files.append(properties_path)
        
        # Generate all Java files
        for item in structure:
            path