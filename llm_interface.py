#!/usr/bin/env python3
"""
Unravel AI - Comprehensive LLM Integration with llama.cpp for CPU inference
"""
import os
import sys
import argparse
import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import huggingface_hub
    HAS_HUGGINGFACE_HUB = True
except ImportError:
    HAS_HUGGINGFACE_HUB = False

@dataclass
class LLMMessage:
    role: str
    content: str

@dataclass
class LLMRequestOptions:
    temperature: float
    max_tokens: int

class LLMProvider(str, Enum):
    LLAMACPP = "llama.cpp"
    # ...existing code for LLMProvider class...

class AnalysisTask(str, Enum):
    # ...existing code for AnalysisTask class...

class InteractionMode(str, Enum):
    """Interaction modes for the LLM"""
    CHAT = "chat"             # General chat interaction
    CODE = "code"             # Code analysis and generation
    SYSTEM = "system"         # System administration tasks
    COMBINED = "combined"     # Multi-purpose interaction

class LLMInteractionManager:
    """Manages different types of LLM interactions"""
    
    def __init__(self, model_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"):
        self.analyzer = CodeAnalyzer(model_id=model_id)
        self.mode = InteractionMode.CHAT
        self.context = []
        self.max_context = 10  # Maximum number of messages to keep
    
    def chat(self, message: str, mode: Optional[InteractionMode] = None) -> str:
        """Interactive chat with context memory"""
        if mode:
            self.mode = mode
        
        # Build system prompt based on mode
        if self.mode == InteractionMode.CHAT:
            system_prompt = "You are a helpful and knowledgeable assistant."
        elif self.mode == InteractionMode.CODE:
            system_prompt = "You are an expert programmer and code analyst."
        elif self.mode == InteractionMode.SYSTEM:
            system_prompt = "You are a system administration expert."
        else:
            system_prompt = "You are a versatile AI assistant capable of general chat, programming, and system administration."
        
        # Build conversation context
        messages = [LLMMessage("system", system_prompt)]
        messages.extend(self.context)
        messages.append(LLMMessage("user", message))
        
        # Generate response
        try:
            response = self.analyzer.api_client.generate(
                messages,
                LLMRequestOptions(
                    temperature=0.7 if self.mode == InteractionMode.CHAT else 0.2,
                    max_tokens=2048
                )
            )
            
            # Update context
            self.context.append(LLMMessage("user", message))
            self.context.append(LLMMessage("assistant", response.content))
            
            # Trim context if too long
            if len(self.context) > self.max_context * 2:
                self.context = self.context[-self.max_context * 2:]
            
            return response.content
        
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Error: {str(e)}"
    
    def clear_context(self):
        """Clear conversation context"""
        self.context = []

# ...existing helper classes...

class LLMModuleManager:
    """Manager for LLM modules and models"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the manager"""
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/unravel-ai/models")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.active_processes = {}
        self.api_clients = {}
            def model_priority(filename):
                if "Q4_K_M" in filename:
                    return 0
                if "Q4_K" in filename:
                    return 1
                if "Q5_K_M" in filename:
                    return 2 
                if "Q5_K" in filename:
                    return 3
                if "Q6_K" in filename:
                    return 4
                if "Q8_0" in filename:
                    return 5
                return 6
            
            # Create model directory
            model_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)
            
            if model_file:
                logger.info(f"Downloading {model_id}/{model_file}...")
                return hf_hub_download(repo_id=model_id, filename=model_file, local_dir=model_dir)
                
            # Try to find a GGUF file
            logger.info(f"Searching for GGUF model in {model_id}...")
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            
            if not gguf_files:
                logger.error(f"No GGUF files found in {model_id}")
                return None
            
            # Sort by size suffix (Q4_K_M is usually good balance)
            def model_priority(filename):
                if "Q4_K_M" in filename: return 0  # noqa: E701
                if "Q4_K" in filename: return 1
                if "Q5_K_M" in filename: return 2 
                if "Q5_K" in filename: return 3
                if "Q6_K" in filename: return 4
                if "Q8_0" in filename: return 5
                return 6
            
            gguf_files.sort(key=model_priority)
            model_file = gguf_files[0]
            
            logger.info(f"Downloading {model_id}/{model_file}...")
            return hf_hub_download(repo_id=model_id, filename=model_file, local_dir=model_dir)
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None

    # ...existing LLMModuleManager methods...

# Modify existing CodeAnalyzer class
class CodeAnalyzer:
    # ...existing CodeAnalyzer code...
    
    def interactive_analysis(self, code: str = "", mode: str = "analysis") -> str:
        """Interactive code analysis session"""
        if not self.is_setup:
            if not self.setup():
                return "Failed to set up analyzer"
        
        if not code:
            return ("Enter code to analyze or use commands:\n"
                   "/mode [analysis|refactor|security|docs] - Change analysis mode\n"
                   "/clear - Clear context\n"
                   "/exit - End session")
        
        if code.startswith("/"):
            cmd = code[1:].lower().split()
            if cmd[0] == "mode" and len(cmd) > 1:
                mode = cmd[1]
                return f"Analysis mode changed to: {mode}"
            elif cmd[0] == "clear":
                return "Context cleared"
            elif cmd[0] == "exit":
                return "Session ended"
            else:
                return "Unknown command"
        
        # Analyze code based on mode
        return self.analyze_code(code, task=mode)

# Create a new CLI class for better interaction
class InteractiveCLI:
    """Interactive CLI for LLM interactions"""
    
    def __init__(self):
        self.interaction = LLMInteractionManager()
        print("Initializing LLM system...")
        if not self.interaction.analyzer.setup():
            print("Failed to initialize system")
            sys.exit(1)
        print("System ready! Type /help for commands")
    
    def run(self):
        """Run interactive CLI"""
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    cmd = user_input[1:].lower().split()
                    if cmd[0] == "help":
                        print("\nCommands:")
                        print("/mode [chat|code|system] - Change interaction mode")
                        print("/clear - Clear conversation context")
                        print("/exit - Exit program")
                        continue
                    elif cmd[0] == "mode" and len(cmd) > 1:
                        try:
                            self.interaction.mode = InteractionMode(cmd[1])
                            print(f"Mode changed to: {cmd[1]}")
                        except ValueError:
                            print("Invalid mode")
                        continue
                    elif cmd[0] == "clear":
                        self.interaction.clear_context()
                        print("Context cleared")
                        continue
                    elif cmd[0] == "exit":
                        print("Goodbye!")
                        break
                
                # Get response
                response = self.interaction.chat(user_input)
                print("\nAssistant:", response)
                
            except KeyboardInterrupt:
                print("\nUse /exit to quit")
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    # Modify main block for CLI options
    parser = argparse.ArgumentParser(description="Unravel AI LLM Module")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--analyze", help="File or directory to analyze")
    # ...rest of existing argument parsing...
    
    args = parser.parse_args()
    
    if args.interactive:
        cli = InteractiveCLI()
        cli.run()
    elif args.analyze:
        # ...existing analysis code...
    else:
        print("Use --interactive for chat mode or --analyze for code analysis")
