#!/usr/bin/env python3
"""
Kaleidoscope AI - Application Generator
=======================================
Transforms natural language app descriptions into full-featured applications.
Integrates with the core Kaleidoscope AI architecture to generate complete,
production-ready codebases from high-level descriptions.
"""

import os
import sys
import json
import logging
import asyncio
import re
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

# Import core Kaleidoscope components
from llm_integration import LLMIntegration, LLMProvider, OpenAIProvider, AnthropicProvider
from error_handling import ErrorManager, ErrorCategory, ErrorSeverity, GracefulDegradation
from code_reusability import UnifiedAST, LanguageAdapterRegistry, MimicryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AppComponent:
    """Represents a component of the application"""
    name: str
    type: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "dependencies": self.dependencies,
            "properties": self.properties,
            "files": self.files
        }

@dataclass
class AppArchitecture:
    """Represents the architecture of an application"""
    name: str
    description: str
    type: str
    language: str
    framework: str
    components: List[AppComponent] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    database: Optional[str] = None
    apis: List[Dict[str, Any]] = field(default_factory=list)
    deployment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "language": self.language,
            "framework": self.framework,
            "components": [c.to_dict() for c in self.components],
            "dependencies": self.dependencies,
            "database": self.database,
            "apis": self.apis,
            "deployment": self.deployment
        }
    
    def add_component(self, component: AppComponent) -> 'AppArchitecture':
        """Add a component to the architecture"""
        self.components.append(component)
        return self

class AppDescriptionAnalyzer:
    """Analyzes app descriptions to extract requirements and architecture"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the analyzer
        
        Args:
            llm_integration: LLM integration for analysis
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
    
    async def analyze_description(self, description: str) -> AppArchitecture:
        """
        Analyze an app description
        
        Args:
            description: App description
            
        Returns:
            Extracted app architecture
        """
        logger.info("Analyzing app description")
        
        # Step 1: Extract basic app properties
        app_props = await self._extract_app_properties(description)
        
        # Step 2: Determine technology stack
        tech_stack = await self._determine_tech_stack(description, app_props)
        
        # Step 3: Identify core components
        components = await self._identify_components(description, app_props, tech_stack)
        
        # Step 4: Build architecture model
        architecture = AppArchitecture(
            name=app_props.get("name", "GeneratedApp"),
            description=app_props.get("description", description),
            type=app_props.get("type", "web"),
            language=tech_stack.get("language", "python"),
            framework=tech_stack.get("framework", ""),
            database=tech_stack.get("database"),
            dependencies=tech_stack.get("dependencies", {}),
            deployment=tech_stack.get("deployment", {})
        )
        
        # Add components
        for comp in components:
            architecture.add_component(AppComponent(
                name=comp["name"],
                type=comp["type"],
                description=comp["description"],
                dependencies=comp.get("dependencies", []),
                properties=comp.get("properties", {})
            ))
        
        return architecture
    
    async def _extract_app_properties(self, description: str) -> Dict[str, Any]:
        """
        Extract basic app properties from description
        
        Args:
            description: App description
            
        Returns:
            Dictionary of app properties
        """
        prompt = f"""
Analyze the following app description and extract key properties.
Return a JSON object with the following fields:
- name: A suitable name for the app
- description: A concise description of the app
- type: The app type (web, mobile, desktop, cli, api, etc.)
- features: List of main features
- target_users: Target user base
- complexity: Estimated complexity (simple, moderate, complex)

App Description:
{description}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            properties = json.loads(response)
            logger.info("Extracted app properties")
            return properties
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="extract_app_properties"
            )
            logger.error(f"Failed to extract app properties: {str(e)}")
            return {
                "name": "GeneratedApp",
                "description": description,
                "type": "web",
                "features": [],
                "target_users": "general",
                "complexity": "moderate"
            }
    
    async def _determine_tech_stack(self, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine appropriate technology stack
        
        Args:
            description: App description
            properties: App properties
            
        Returns:
            Dictionary of technology stack details
        """
        prompt = f"""
Based on the following app description and properties, recommend an appropriate technology stack.
Return a JSON object with the following fields:
- language: Primary programming language
- framework: Main framework to use
- database: Database technology (if needed)
- frontend: Frontend framework/library (if applicable)
- backend: Backend technology (if applicable)
- mobile: Mobile development approach (if applicable)
- dependencies: Key libraries and dependencies
- deployment: Recommended deployment approach

App Description:
{description}

App Properties:
{json.dumps(properties, indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            tech_stack = json.loads(response)
            logger.info(f"Determined tech stack: {tech_stack['language']}/{tech_stack['framework']}")
            return tech_stack
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="determine_tech_stack"
            )
            logger.error(f"Failed to determine tech stack: {str(e)}")
            return {
                "language": "python",
                "framework": "flask",
                "database": "sqlite",
                "frontend": "bootstrap",
                "backend": "rest",
                "dependencies": {},
                "deployment": {"type": "docker"}
            }
    
    async def _identify_components(
        self, 
        description: str, 
        properties: Dict[str, Any], 
        tech_stack: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify core components of the app
        
        Args:
            description: App description
            properties: App properties
            tech_stack: Technology stack
            
        Returns:
            List of component dictionaries
        """
        prompt = f"""
Identify the core components needed for this application.
Return a JSON array of component objects, each with:
- name: Component name
- type: Component type (model, view, controller, service, utility, etc.)
- description: What the component does
- dependencies: List of other components this depends on
- properties: Additional component-specific properties

App Description:
{description}

App Properties:
{json.dumps(properties, indent=2)}

Technology Stack:
{json.dumps(tech_stack, indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            components = json.loads(response)
            logger.info(f"Identified {len(components)} components")
            return components
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="identify_components"
            )
            logger.error(f"Failed to identify components: {str(e)}")
            
            # Return some basic components based on tech stack
            basic_components = []
            
            if tech_stack.get("language") == "python":
                if tech_stack.get("framework") == "flask":
                    basic_components = [
                        {
                            "name": "app",
                            "type": "application",
                            "description": "Main Flask application",
                            "dependencies": []
                        },
                        {
                            "name": "models",
                            "type": "model",
                            "description": "Data models",
                            "dependencies": []
                        },
                        {
                            "name": "views",
                            "type": "view",
                            "description": "Web views and routes",
                            "dependencies": ["models"]
                        }
                    ]
                elif tech_stack.get("framework") == "django":
                    basic_components = [
                        {
                            "name": "app",
                            "type": "application",
                            "description": "Main Django application",
                            "dependencies": []
                        },
                        {
                            "name": "models",
                            "type": "model",
                            "description": "Django models",
                            "dependencies": []
                        },
                        {
                            "name": "views",
                            "type": "view",
                            "description": "Django views",
                            "dependencies": ["models"]
                        },
                        {
                            "name": "urls",
                            "type": "router",
                            "description": "URL routing",
                            "dependencies": ["views"]
                        }
                    ]
            elif tech_stack.get("language") == "javascript" or tech_stack.get("language") == "typescript":
                if tech_stack.get("framework") == "react":
                    basic_components = [
                        {
                            "name": "App",
                            "type": "component",
                            "description": "Main React application component",
                            "dependencies": []
                        },
                        {
                            "name": "components",
                            "type": "ui",
                            "description": "UI components",
                            "dependencies": []
                        },
                        {
                            "name": "services",
                            "type": "service",
                            "description": "API services",
                            "dependencies": []
                        }
                    ]
                elif tech_stack.get("framework") == "express":
                    basic_components = [
                        {
                            "name": "app",
                            "type": "application",
                            "description": "Express application",
                            "dependencies": []
                        },
                        {
                            "name": "routes",
                            "type": "router",
                            "description": "API routes",
                            "dependencies": []
                        },
                        {
                            "name": "models",
                            "type": "model",
                            "description": "Data models",
                            "dependencies": []
                        }
                    ]
            
            return basic_components

class CodeGenerator:
    """Generates code for application components"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the code generator
        
        Args:
            llm_integration: LLM integration for code generation
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
    
    async def generate_component_code(
        self, 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> Dict[str, str]:
        """
        Generate code for a component
        
        Args:
            component: App component
            architecture: App architecture
            
        Returns:
            Dictionary mapping file paths to code
        """
        logger.info(f"Generating code for component: {component.name}")
        
        # Determine files needed for this component
        files = await self._determine_component_files(component, architecture)
        
        # Generate code for each file
        results = {}
        for file_info in files:
            file_path = file_info["path"]
            try:
                code = await self._generate_file_code(file_path, file_info, component, architecture)
                results[file_path] = code
                logger.info(f"Generated code for {file_path}")
            except Exception as e:
                error = self.error_manager.handle_exception(
                    e, 
                    category=ErrorCategory.GENERATION,
                    severity=ErrorSeverity.ERROR,
                    operation="generate_component_code",
                    component=component.name,
                    file_path=file_path
                )
                logger.error(f"Failed to generate code for {file_path}: {str(e)}")
                
                # Add a placeholder file with error information
                results[file_path] = f"""
# Error generating this file
# {str(e)}
# Please regenerate this file
"""
        
        return results
    
    async def _determine_component_files(
        self, 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> List[Dict[str, Any]]:
        """
        Determine files needed for a component
        
        Args:
            component: App component
            architecture: App architecture
            
        Returns:
            List of file information dictionaries
        """
        prompt = f"""
Determine the files needed for this component in the application.
Return a JSON array of file objects, each with:
- path: Relative file path
- purpose: Purpose of this file
- dependencies: Other files this depends on
- content_type: Type of content (code, config, static, etc.)

Component:
{json.dumps(component.to_dict(), indent=2)}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            files = json.loads(response)
            return files
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.ANALYSIS,
                severity=ErrorSeverity.ERROR,
                operation="determine_component_files"
            )
            logger.error(f"Failed to determine component files: {str(e)}")
            
            # Return a basic file for the component
            language = architecture.language.lower()
            extension = self._get_extension_for_language(language)
            
            if component.type == "model":
                return [{
                    "path": f"{component.name}{extension}",
                    "purpose": "Define data models",
                    "dependencies": [],
                    "content_type": "code"
                }]
            elif component.type == "controller" or component.type == "view":
                return [{
                    "path": f"{component.name}{extension}",
                    "purpose": f"Implement {component.type}",
                    "dependencies": [],
                    "content_type": "code"
                }]
            else:
                return [{
                    "path": f"{component.name}{extension}",
                    "purpose": component.description,
                    "dependencies": [],
                    "content_type": "code"
                }]
    
    def _get_extension_for_language(self, language: str) -> str:
        """Get the file extension for a language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "c#": ".cs",
            "ruby": ".rb",
            "php": ".php",
            "go": ".go",
            "rust": ".rs",
            "c++": ".cpp",
            "c": ".c"
        }
        return extensions.get(language.lower(), ".txt")
    
    async def _generate_file_code(
        self, 
        file_path: str, 
        file_info: Dict[str, Any], 
        component: AppComponent, 
        architecture: AppArchitecture
    ) -> str:
        """
        Generate code for a file
        
        Args:
            file_path: File path
            file_info: File information
            component: App component
            architecture: App architecture
            
        Returns:
            Generated code
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self._get_language_for_extension(file_ext) or architecture.language
        
        prompt = f"""
Generate complete, production-ready code for this file in {language}.
The code should be fully functional, well-structured, and follow best practices.
Include comprehensive error handling, logging, validation, and comments.
Do not use placeholder functions or methods - implement everything completely.

File Path: {file_path}
Purpose: {file_info['purpose']}

Component:
{json.dumps(component.to_dict(), indent=2)}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Generate only the code for this file, with no additional explanation.
"""
        response = await self.llm.generate_completion(prompt)
        
        # Clean up the response
        code = self._clean_generated_code(response, language)
        
        return code
    
    def _get_language_for_extension(self, ext: str) -> Optional[str]:
        """Get the language for a file extension"""
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "react",
            ".tsx": "react-typescript",
            ".java": "java",
            ".cs": "c#",
            ".rb": "ruby",
            ".php": "php",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "c++",
            ".c": "c",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".md": "markdown",
            ".sh": "bash"
        }
        return extensions.get(ext)
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """
        Clean up generated code
        
        Args:
            code: Generated code
            language: Programming language
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks if present
        code = re.sub(r'```[a-z]*\n', '', code)
        code = re.sub(r'```\n?$', '', code)
        
        # Remove unnecessary comments about implementation
        code = re.sub(r'# TODO: Implement.*?\n', '', code)
        code = re.sub(r'// TODO: Implement.*?\n', '', code)
        
        # Ensure proper line endings
        code = code.replace('\r\n', '\n')
        
        return code.strip()

class AppStructureGenerator:
    """Generates the overall application structure"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize the app structure generator
        
        Args:
            llm_integration: LLM integration for structure generation
        """
        self.llm = llm_integration
        self.error_manager = ErrorManager()
        self.code_generator = CodeGenerator(llm_integration)
    
    async def generate_app_structure(
        self, 
        architecture: AppArchitecture, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Generate the complete app structure
        
        Args:
            architecture: App architecture
            output_dir: Output directory
            
        Returns:
            Generation results
        """
        logger.info(f"Generating app structure for {architecture.name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate project structure information
        structure = await self._generate_project_structure(architecture)
        
        # Create directories
        for dir_path in structure.get("directories", []):
            full_path = os.path.join(output_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Generate configuration files
        config_files = await self._generate_config_files(architecture, structure)
        
        # Write configuration files
        for file_path, content in config_files.items():
            full_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            logger.info(f"Created configuration file: {file_path}")
        
        # Generate code for each component
        component_files = {}
        for component in architecture.components:
            files = await self.code_generator.generate_component_code(component, architecture)
            component_files[component.name] = files
            
            # Write component files
            for file_path, content in files.items():
                full_path = os.path.join(output_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created component file: {file_path}")
        
        # Generate README and documentation
        docs = await self._generate_documentation(architecture, structure)
        
        # Write documentation files
        for file_path, content in docs.items():
            full_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            logger.info(f"Created documentation file: {file_path}")
        
        # Return results
        return {
            "structure": structure,
            "config_files": config_files,
            "component_files": component_files,
            "documentation": docs
        }
    
    async def _generate_project_structure(self, architecture: AppArchitecture) -> Dict[str, Any]:
        """
        Generate project structure information
        
        Args:
            architecture: App architecture
            
        Returns:
            Project structure information
        """
        prompt = f"""
Generate the complete project structure for this application.
Return a JSON object with:
- directories: Array of directory paths to create
- root_files: Array of files in the root directory
- standard_files: Common files for this technology stack
- entry_point: Main entry point file
- package_file: Package management file

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}
"""
        try:
            response = await self.llm.generate_completion(prompt)
            structure = json.loads(response)
            logger.info(f"Generated project structure with {len(structure.get('directories', []))} directories")
            return structure
        except Exception as e:
            error = self.error_manager.handle_exception(
                e, 
                category=ErrorCategory.GENERATION,
                severity=ErrorSeverity.ERROR,
                operation="generate_project_structure"
            )
            logger.error(f"Failed to generate project structure: {str(e)}")
            
            # Return a basic structure based on the framework
            framework = architecture.framework.lower()
            language = architecture.language.lower()
            
            if language == "python":
                if framework == "flask":
                    return {
                        "directories": ["app", "app/models", "app/routes", "app/templates", "app/static", "tests"],
                        "root_files": ["requirements.txt", "app.py", ".gitignore", "README.md"],
                        "standard_files": ["app/__init__.py", "app/models/__init__.py", "app/routes/__init__.py"],
                        "entry_point": "app.py",
                        "package_file": "requirements.txt"
                    }
                elif framework == "django":
                    app_name = architecture.name.lower().replace(" ", "_")
                    return {
                        "directories": [app_name, f"{app_name}/core", f"{app_name}/templates", f"{app_name}/static", "tests"],
                        "root_files": ["requirements.txt", "manage.py", ".gitignore", "README.md"],
                        "standard_files": [f"{app_name}/__init__.py", f"{app_name}/settings.py", f"{app_name}/urls.py", f"{app_name}/wsgi.py"],
                        "entry_point": "manage.py",
                        "package_file": "requirements.txt"
                    }
            elif language == "javascript" or language == "typescript":
                is_ts = language == "typescript"
                ext = ".ts" if is_ts else ".js"
                
                if framework == "react":
                    return {
                        "directories": ["src", "src/components", "src/services", "src/styles", "public"],
                        "root_files": ["package.json", ".gitignore", "README.md", "tsconfig.json" if is_ts else ""],
                        "standard_files": [f"src/index{ext}", f"src/App{ext}", "public/index.html"],
                        "entry_point": f"src/index{ext}",
                        "package_file": "package.json"
                    }
                elif framework == "express":
                    return {
                        "directories": ["src", "src/routes", "src/models", "src/controllers", "src/middleware", "tests"],
                        "root_files": ["package.json", ".gitignore", "README.md", "tsconfig.json" if is_ts else ""],
                        "standard_files": [f"src/index{ext}", f"src/app{ext}"],
                        "entry_point": f"src/index{ext}",
                        "package_file": "package.json"
                    }
            
            # Generic structure
            return {
                "directories": ["src", "tests", "docs"],
                "root_files": ["README.md", ".gitignore"],
                "standard_files": ["src/main.py" if language == "python" else "src/index.js"],
                "entry_point": "src/main.py" if language == "python" else "src/index.js",
                "package_file": "requirements.txt" if language == "python" else "package.json"
            }
    
    async def _generate_config_files(
        self, 
        architecture: AppArchitecture, 
        structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate configuration files
        
        Args:
            architecture: App architecture
            structure: Project structure information
            
        Returns:
            Dictionary mapping file paths to content
        """
        config_files = {}
        
        # Generate each standard file
        for file_path in structure.get("standard_files", []) + structure.get("root_files", []):
            if not file_path:  # Skip empty entries
                continue
                
            prompt = f"""
Generate complete, production-ready content for this configuration file.
The content should be properly formatted and follow best practices for this file type.

File Path: {file_path}

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Generate only the file content, with no additional explanation.
"""
            try:
                content = await self.llm.generate_completion(prompt)
                
                # Clean up the content
                content = self._clean_file_content(content, file_path)
                
                config_files[file_path] = content
                logger.info(f"Generated content for {file_path}")
            except Exception as e:
                error = self.error_manager.handle_exception(
                    e, 
                    category=ErrorCategory.GENERATION,
                    severity=ErrorSeverity.ERROR,
                    operation="generate_config_file",
                    file_path=file_path
                )
                logger.error(f"Failed to generate content for {file_path}: {str(e)}")
                
                # Add a placeholder file with error information
                config_files[file_path] = f"""
# Error generating this file
# {str(e)}
# Please regenerate this file
"""
        
        return config_files
    
    def _clean_file_content(self, content: str, file_path: str) -> str:
        """
        Clean up generated file content
        
        Args:
            content: Generated content
            file_path: File path
            
        Returns:
            Cleaned content
        """
        # Remove markdown code blocks if present
        content = re.sub(r'```[a-z]*\n', '', content)
        content = re.sub(r'```\n?$', '', content)
        
        # Ensure proper line endings
        content = content.replace('\r\n', '\n')
        
        # Special handling for specific file types
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # Ensure valid JSON
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # If it's not valid JSON, leave it as is
                pass
        
        return content.strip()
    
    async def _generate_documentation(
        self, 
        architecture: AppArchitecture, 
        structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate documentation files
        
        Args:
            architecture: App architecture
            structure: Project structure information
            
        Returns:
            Dictionary mapping file paths to content
        """
        docs = {}
        
        # Generate README
        prompt = f"""
Generate a comprehensive README.md file for this application.
Include sections for:
1. Project overview and description
2. Features
3. Technology stack
4. Installation instructions
5. Usage/getting started
6. Project structure
7. API documentation (if applicable)
8. Testing
9. Deployment
10. License information

Application Architecture:
{json.dumps(architecture.to_dict(), indent=2)}

Project Structure:
{json.dumps(structure, indent=2)}

Generate only the README content, formatted in Markdown.
"""
        try:
            readme_content = await self.llm.generate_completion(prompt)
            docs["README.md"] = readme_content
            
