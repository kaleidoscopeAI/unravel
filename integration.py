#!/usr/bin/env python3
"""
Unravel AI - Integrated Application Generator
==================================================
A system that converts natural language descriptions into
fully functional applications using quantum consciousness
models and advanced code generation techniques.
"""

import os
import sys
import logging
import asyncio
import time
import json
import hashlib
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Import the core components
from artificialThinker import ConsciousController, QuantumConsciousnessAPI
from app_generator import AppDescriptionAnalyzer, CodeGenerator, AppStructureGenerator, AppComponent, AppArchitecture
from execution_sandbox_system import DockerSandbox, SandboxConfig, SandboxManager
from ingestion import TokenCounter, FileAnalyzer, SpecGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("unravel_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LLMIntegration:
    """Integration with Language Models for code generation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.token_counter = TokenCounter(model)
        
    async def generate_completion(self, prompt: str) -> str:
        """Generate completion using the LLM"""
        # In a real implementation, this would call an API
        # For now, we'll use the ConsciousController to generate responses
        controller = ConsciousController()
        result = controller.process_text(prompt, "system")
        return result["response"]

class InspirationCrawler:
    """Crawls repositories for code inspiration"""
    
    def __init__(self, token_counter=None):
        self.token_counter = token_counter or TokenCounter()
        self.file_analyzer = FileAnalyzer()
        
    async def find_inspiration(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find repositories and code snippets matching requirements"""
        # Extract key concepts from requirements
        concepts = []
        if "name" in requirements:
            concepts.extend(requirements["name"].lower().split())
        if "description" in requirements:
            concepts.extend(requirements["description"].lower().split())
        if "type" in requirements:
            concepts.append(requirements["type"])
        if "language" in requirements:
            concepts.append(requirements["language"])
        if "framework" in requirements:
            concepts.append(requirements["framework"])
        
        # Get unique concepts
        concepts = list(set([c for c in concepts if len(c) > 3]))
        
        # In a real implementation, this would search GitHub or other repositories
        # For now, we'll just return dummy snippets
        inspirations = []
        for i, concept in enumerate(concepts[:3]):
            inspirations.append({
                "name": f"example_{concept}",
                "language": requirements.get("language", "python"),
                "snippet": f"# Example code for {concept}\ndef {concept}_function():\n    return 'Implementation for {concept}'",
                "url": f"https://github.com/example/{concept}",
                "relevance": 0.9 - (i * 0.1)
            })
        
        return inspirations

class UnravelIntegrator:
    """Main integrator class that ties all components together"""
    
    def __init__(self):
        """Initialize the Unravel AI integrator"""
        # Core components
        self.llm = LLMIntegration()
        self.consciousness = ConsciousController(dimension=4, resolution=64)
        self.api = QuantumConsciousnessAPI(dimension=4, resolution=64)
        self.crawler = InspirationCrawler()
        
        # App generation components
        self.app_analyzer = AppDescriptionAnalyzer(self.llm)
        self.code_generator = CodeGenerator(self.llm)
        self.structure_generator = AppStructureGenerator(self.llm)
        
        # Testing components
        self.sandbox_manager = SandboxManager()
        
        # State
        self.conversation_history = []
        self.generated_apps = {}
        self.active_sandboxes = {}
        self.knowledge_base = []
        
        # Create output directories
        os.makedirs("generated", exist_ok=True)
        os.makedirs("knowledge", exist_ok=True)
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Determine if this is an app generation request
        is_app_request = self._is_app_generation_request(message)
        
        # Process through the consciousness system either way
        consciousness_result = await self.api.chat(message)
        
        # Store the consciousness response
        consciousness_response = consciousness_result["response"]
        
        if is_app_request:
            # Generate app
            app_result = await self._generate_app(message, consciousness_response)
            
            # Format response including app details
            response = self._format_app_response(app_result)
        else:
            # Just return the consciousness response for regular chat
            response = consciousness_response
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Return complete results
        return {
            "response": response,
            "consciousness_state": consciousness_result["system_state"],
            "app_generated": is_app_request,
            "app_details": app_result if is_app_request else None
        }
    
    def _is_app_generation_request(self, message: str) -> bool:
        """Determine if a message is requesting app generation"""
        # Look for common patterns that indicate app generation requests
        app_indicators = [
            "create an app", "build an application", "make a program",
            "develop a", "generate a", "code a", "implement a",
            "app that", "application for", "program for", "website for",
            "can you build", "can you create", "can you make"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in app_indicators)
    
    async def _generate_app(self, message: str, consciousness_response: str) -> Dict[str, Any]:
        """Generate an application based on user message"""
        logger.info(f"Generating application from message: {message[:50]}...")
        
        # Step 1: Analyze app description and extract architecture
        architecture = await self.app_analyzer.analyze_description(message)
        
        # Step 2: Find inspiration from similar projects
        inspiration = await self.crawler.find_inspiration(architecture.to_dict())
        
        # Step 3: Generate code for each component
        components_code = {}
        for component in architecture.components:
            try:
                code = await self.code_generator.generate_component_code(component, architecture)
                components_code[component.name] = code
            except Exception as e:
                logger.error(f"Error generating code for component {component.name}: {str(e)}")
                components_code[component.name] = {"error": str(e)}
        
        # Step 4: Generate application structure
        output_dir = f"generated/{architecture.name.lower().replace(' ', '_')}"
        structure_result = await self.structure_generator.generate_app_structure(
            architecture, output_dir
        )
        
        # Step 5: Test in sandbox if needed
        sandbox_result = None
        if os.path.exists(output_dir):
            try:
                # Create sandbox configuration
                sandbox_config = SandboxConfig(
                    app_dir=output_dir,
                    app_name=architecture.name,
                    app_type=architecture.type,
                    language=architecture.language,
                    framework=architecture.framework,
                    auto_open_browser=False  # Don't open browser automatically
                )
                
                # Run in sandbox
                sandbox_result = self.sandbox_manager.create_sandbox(
                    sandbox_config.app_dir, sandbox_config.__dict__
                )
                
                # Store active sandbox
                self.active_sandboxes[architecture.name] = sandbox_result
            except Exception as e:
                logger.error(f"Error running app in sandbox: {str(e)}")
                sandbox_result = {"error": str(e)}
        
        # Store generated app info
        app_id = hashlib.md5(architecture.name.encode()).hexdigest()[:12]
        self.generated_apps[app_id] = {
            "id": app_id,
            "name": architecture.name,
            "description": architecture.description,
            "path": output_dir,
            "architecture": architecture.to_dict(),
            "generated_at": time.time(),
            "sandbox": sandbox_result
        }
        
        # Return results
        return {
            "id": app_id,
            "name": architecture.name,
            "description": architecture.description,
            "architecture": architecture.to_dict(),
            "components": {name: list(files.keys()) for name, files in components_code.items()},
            "structure": structure_result,
            "inspiration": inspiration,
            "sandbox": sandbox_result,
            "output_dir": output_dir
        }
    
    def _format_app_response(self, app_result: Dict[str, Any]) -> str:
        """Format a response about the generated application"""
        # Create a user-friendly response
        app_name = app_result["name"]
        app_type = app_result["architecture"]["type"]
        language = app_result["architecture"]["language"]
        framework = app_result["architecture"]["framework"]
        
        # Format component list
        components = []
        for component_name, files in app_result["components"].items():
            num_files = len(files)
            components.append(f"- {component_name} ({num_files} file{'s' if num_files != 1 else ''})")
        
        # Format sandbox information
        sandbox_info = ""
        if app_result["sandbox"] and "error" not in app_result["sandbox"]:
            urls = app_result["sandbox"].get("urls", {})
            if urls:
                sandbox_info = "\n\nYour application is running at:\n"
                for port, url in urls.items():
                    sandbox_info += f"- {url}\n"
            else:
                sandbox_info = "\n\nYour application is running in the sandbox."
        
        # Build complete response
        response = f"""I've created your {app_type} application "{app_name}" using {language} with {framework}.

The application has been generated with the following components:
{"".join(f"{c}\n" for c in components)}

The code has been saved to {app_result["output_dir"]}.{sandbox_info}

Would you like me to explain any specific part of the application in more detail?"""
        
        return response
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add knowledge to the system"""
        # Process through consciousness system
        result = await self.api.add_knowledge(content, metadata or {})
        
        # Store in knowledge base
        knowledge_id = result["memory_id"]
        knowledge_entry = {
            "id": knowledge_id,
            "content": content,
            "metadata": metadata or {},
            "added_at": time.time()
        }
        self.knowledge_base.append(knowledge_entry)
        
        # Save to file for persistence
        knowledge_file = f"knowledge/{knowledge_id}.json"
        with open(knowledge_file, 'w') as f:
            json.dump(knowledge_entry, f, indent=2)
        
        return {
            "success": True,
            "knowledge_id": knowledge_id,
            "node_id": result["node_id"]
        }
    
    def get_app_details(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a generated app"""
        return self.generated_apps.get(app_id)
    
    def get_sandbox_status(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a sandbox"""
        sandbox_id = self.active_sandboxes.get(app_name, {}).get("sandbox_id")
        if sandbox_id:
            # Get current status
            return self.sandbox_manager.get_sandbox_status(sandbox_id)
        return None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return self.consciousness.get_state()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

class UnravelCLI:
    """Command-line interface for the Unravel AI system"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.integrator = UnravelIntegrator()
        
    async def start_interactive(self):
        """Start interactive CLI session"""
        print("Unravel AI - Interactive Console")
        print("Type 'exit' to quit, 'help' for commands")
        print()
        
        # Main loop
        while True:
            try:
                # Get user input
                message = input("You: ")
                
                # Handle special commands
                if message.lower() == 'exit':
                    break
                elif message.lower() == 'help':
                    self._show_help()
                    continue
                elif message.lower().startswith('status'):
                    # Show system status
                    await self._show_status()
                    continue
                elif message.lower().startswith('apps'):
                    # List generated apps
                    self._list_apps()
                    continue
                elif message.lower().startswith('sandbox'):
                    # Show sandbox status
                    await self._show_sandbox(message)
                    continue
                
                # Process regular message
                result = await self.integrator.process_message(message)
                
                # Print response
                print("\nUnravel: " + result["response"])
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _show_help(self):
        """Show help information"""
        print("\nUnravel AI Commands:")
        print("  help        - Show this help message")
        print("  exit        - Exit the program")
        print("  status      - Show system status")
        print("  apps        - List generated applications")
        print("  sandbox APP - Show sandbox status for APP")
        print()
    
    async def _show_status(self):
        """Show system status"""
        consciousness = await self.integrator.api.get_consciousness_state()
        
        print("\nSystem Status:")
        print(f"  Consciousness Level: {consciousness['consciousness_level']:.3f}")
        print(f"  Active Nodes: {consciousness['nodes_count']}")
        print(f"  Generated Apps: {len(self.integrator.generated_apps)}")
        print(f"  Active Sandboxes: {len(self.integrator.active_sandboxes)}")
        print(f"  Knowledge Items: {len(self.integrator.knowledge_base)}")
        print(f"  Conversation Length: {len(self.integrator.conversation_history)}")
        print()
    
    def _list_apps(self):
        """List generated applications"""
        apps = self.integrator.generated_apps
        
        if not apps:
            print("\nNo applications generated yet.")
            print()
            return
        
        print("\nGenerated Applications:")
        for app_id, app in apps.items():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(app["generated_at"]))
            print(f"  {app['name']} (ID: {app_id})")
            print(f"    Type: {app['architecture']['type']}")
            print(f"    Language: {app['architecture']['language']}")
            print(f"    Path: {app['path']}")
            print(f"    Generated: {timestamp}")
            
            # Show sandbox status if available
            if app.get('sandbox') and 'error' not in app['sandbox']:
                status = "Running"
            else:
                status = "Not running"
            print(f"    Sandbox: {status}")
            print()
    
    async def _show_sandbox(self, message):
        """Show sandbox status for a specific app"""
        parts = message.split(maxsplit=1)
        if len(parts) < 2:
            print("\nUsage: sandbox APP_NAME")
            print()
            return
        
        app_name = parts[1]
        
        # Find app by name
        app_id = None
        for id, app in self.integrator.generated_apps.items():
            if app['name'].lower() == app_name.lower():
                app_id = id
                break
        
        if not app_id:
            print(f"\nNo application found with name '{app_name}'.")
            print()
            return
        
        app = self.integrator.generated_apps[app_id]
        
        # Get sandbox status
        status = self.integrator.get_sandbox_status(app['name'])
        
        print(f"\nSandbox Status for {app['name']}:")
        if status:
            print(f"  Status: {status.get('status', 'Unknown')}")
            print(f"  Container: {status.get('container_name', 'Unknown')}")
            
            # Show URLs if available
            urls = status.get('urls', {})
            if urls:
                print("  URLs:")
                for port, url in urls.items():
                    print(f"    {url}")
        else:
            print("  Not running")
        print()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unravel AI - Application Generator")
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive console')
    parser.add_argument('--generate', '-g', type=str, help='Generate app from description')
    
    args = parser.parse_args()
    
    if args.interactive:
        cli = UnravelCLI()
        await cli.start_interactive()
    elif args.generate:
        integrator = UnravelIntegrator()
        result = await integrator.process_message(args.generate)
        print(result["response"])
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
