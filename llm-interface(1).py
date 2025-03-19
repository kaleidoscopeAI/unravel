#!/usr/bin/env python3
"""
Unravel AI - Local LLM Integration Module
Interfaces with open-source Language Models to enhance code analysis, specification generation,
and software reconstruction capabilities, with focus on Hugging Face models.

Usage examples:
    
    # Basic usage with Hugging Face model
    model_config = ModelConfig(
        model_id="codellama/CodeLlama-7b-Instruct-hf",
        provider=LocalModelProvider.HUGGINGFACE,
        load_in_4bit=True
    )
    llm = LocalLLMInterface(model_config)
    
    # Prepare messages
    messages = [
        LLMMessage(role="system", content="You are a helpful coding assistant specialized in Python."),
        LLMMessage(role="user", content="What's a good way to implement a binary search tree in Python?")
    ]
    
    # Generate response
    response = llm.generate(messages)
    print(response.content)
    
    # Using Ollama
    model_config = ModelConfig(
        model_id="codellama:7b-instruct",
        provider=LocalModelProvider.OLLAMA
    )
    llm = LocalLLMInterface(model_config)
    response = llm.generate(messages)
    
    # Using with Unravel AI's analysis pipeline
    def analyze_with_llm(decompiled_files, llm_interface):
        # Process files and extract key information
        file_contents = {}
        for file_path in decompiled_files:
            with open(file_path, 'r', errors='ignore') as f:
                file_contents[file_path] = f.read()
        
        # Create prompt for the LLM
        prompt = "Analyze the following code and provide insights:\\n\\n"
        for path, content in file_contents.items():
            prompt += f"File: {os.path.basename(path)}\\n```\\n{content}\\n```\\n\\n"
        
        # Add specific questions
        prompt += "What are the main functions and classes? What algorithms are being used? Are there any security vulnerabilities?"
        
        # Get LLM response
        messages = [
            LLMMessage(role="system", content="You are an expert code analyzer for the Unravel AI system."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = llm_interface.generate(messages)
        return response.content
"""

import os
import sys
import json
import time
import asyncio
import logging
import aiohttp
import tiktoken
import hashlib
import base64
import torch
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
import networkx as nx
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download, login
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ModelSize(Enum):
    """Model size categories based on parameter count"""
    SMALL = "small"      # 1B-3B parameters
    MEDIUM = "medium"    # 3B-7B parameters
    LARGE = "large"      # 7B-13B parameters
    XLARGE = "xlarge"    # 13B-33B parameters
    XXLARGE = "xxlarge"  # 33B+ parameters

class LocalModelProvider(Enum):
    """Supported open-source model providers"""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"

@dataclass
class ModelConfig:
    """Configuration for a local model"""
    model_id: str
    provider: LocalModelProvider = LocalModelProvider.HUGGINGFACE
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    revision: str = "main"
    size: ModelSize = ModelSize.MEDIUM
    trust_remote_code: bool = False
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    cpu_only: bool = False
    device_map: str = "auto"
    max_memory: Optional[Dict[int, str]] = None  # Device ID to memory limit
    offload_folder: Optional[str] = None
    use_cache: bool = True
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype"""
        if self.torch_dtype == "auto":
            return "auto"
        elif self.torch_dtype == "float16":
            return torch.float16
        elif self.torch_dtype == "bfloat16":
            return torch.bfloat16
        elif self.torch_dtype == "float32":
            return torch.float32
        else:
            return "auto"

@dataclass
class LLMMessage:
    """Represents a message in a conversation with an LLM"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API requests"""
        return {"role": self.role, "content": self.content}

@dataclass
class LLMRequestOptions:
    """Options for an LLM request"""
    temperature: float = 0.2
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    seed: Optional[int] = None
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    num_return_sequences: int = 1

@dataclass
class LLMUsage:
    """Tracks token usage"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResponse:
    """Response from an LLM"""
    content: str
    usage: LLMUsage
    model_id: str
    finish_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "usage": asdict(self.usage),
            "model_id": self.model_id,
            "finish_reason": self.finish_reason
        }

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for text generation"""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        # Pre-tokenize stop sequences for faster checking
        self.stop_sequences_ids = [
            tokenizer.encode(seq, add_special_tokens=False) 
            for seq in stop_sequences
        ]
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Check if most recently generated token sequence matches any stop sequence
        for stop_ids in self.stop_sequences_ids:
            if len(stop_ids) == 0:
                continue
                
            # Get the last n tokens where n is the length of the longest stop sequence
            last_tokens = input_ids[0, -(len(stop_ids)):]
            
            # Check if these tokens match the stop sequence
            if torch.all(last_tokens == torch.tensor(stop_ids, device=input_ids.device)):
                return True
                
        return False

class TokenCounter:
    """Utility for counting tokens in LLM requests"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.tiktoken_encoder = None
        
        # Try to use tiktoken if available for faster counting with OpenAI-compatible tokenization
        try:
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("tiktoken not available, falling back to Hugging Face tokenizer")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if self.tiktoken_encoder:
            return len(self.tiktoken_encoder.encode(text))
        elif self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate if no tokenizer available - very approximate!
            return len(text.split())
    
    def count_message_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in a list of messages"""
        token_count = 0
        
        for message in messages:
            # Add message metadata tokens (approximate)
            token_count += 4
            
            # Add content tokens
            token_count += self.count_tokens(message.content)
        
        # Add extra tokens for the message format
        token_count += 2
        
        return token_count

class LocalLLMInterface:
    """Interface for running inference using local LLMs"""
    
    async def generate_async(self, messages: List[LLMMessage], options: Optional[LLMRequestOptions] = None) -> LLMResponse:
        """
        Generate a response from the LLM asynchronously
        
        Args:
            messages: List of conversation messages
            options: Generation options
            
        Returns:
            LLM response
        """
        # Currently, we just call the synchronous version in a thread pool
        # This could be optimized in the future for truly async operation
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                lambda: self.generate(messages, options)
            )
        return result
    
    async def generate_stream_async(self, messages: List[LLMMessage], options: Optional[LLMRequestOptions] = None) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM asynchronously
        
        Args:
            messages: List of conversation messages
            options: Generation options
            
        Yields:
            Chunks of the generated response
        """
        if options is None:
            options = LLMRequestOptions(stream=True)
        else:
            options.stream = True
        
        # For Hugging Face models, we need special handling
        if self.model_config.provider == LocalModelProvider.HUGGINGFACE:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Format prompt based on model type
            prompt = self._format_prompt(messages)
            
            # Set up streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            
            # Configure generation parameters
            generation_config = {
                "max_new_tokens": options.max_tokens or 2048,
                "temperature": options.temperature,
                "top_p": options.top_p,
                "top_k": options.top_k,
                "repetition_penalty": options.repetition_penalty,
                "do_sample": options.temperature > 0.0,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Set up stopping criteria if needed
            if options.stop_sequences:
                custom_stopping_criteria = CustomStoppingCriteria(options.stop_sequences, self.tokenizer)
                stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
                generation_config["stopping_criteria"] = stopping_criteria
            
            # Start generation in a separate thread
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                generation_task = loop.run_in_executor(
                    executor,
                    lambda: self.pipeline(prompt, **generation_config)
                )
                
                # Stream the response
                for text in streamer:
                    yield text
                
                # Ensure generation completes
                await generation_task
        else:
            # For API-based providers, we use their streaming capability
            # But we implement it as a synchronous operation wrapped in a thread
            chunks_queue = asyncio.Queue()
            
            async def _stream_worker():
                if self.model_config.provider == LocalModelProvider.OLLAMA:
                    await self._stream_ollama(messages, options, chunks_queue)
                elif self.model_config.provider == LocalModelProvider.VLLM:
                    await self._stream_vllm(messages, options, chunks_queue)
                elif self.model_config.provider == LocalModelProvider.LLAMACPP:
                    await self._stream_llamacpp(messages, options, chunks_queue)
                # Signal completion
                await chunks_queue.put(None)
            
            # Start streaming task
            task = asyncio.create_task(_stream_worker())
            
            # Yield chunks as they become available
            while True:
                chunk = await chunks_queue.get()
                if chunk is None:  # End of stream
                    break
                yield chunk
            
            # Ensure task completes
            await task
    
    def __init__(self, 
                model_config: ModelConfig,
                cache_dir: Optional[str] = None,
                hf_token: Optional[str] = None,
                inference_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the Local LLM interface
        
        Args:
            model_config: Configuration for the model
            cache_dir: Directory to cache model files and responses
            hf_token: Hugging Face API token for downloading models
            inference_settings: Additional inference settings
        """
        self.model_config = model_config
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/unravel-ai/models")
        self.hf_token = hf_token
        self.inference_settings = inference_settings or {}
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize model and tokenizer to None (will be loaded on demand)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Set provider-specific paths
        self._setup_model_paths()
        
        # Log in to Hugging Face if token provided
        if self.hf_token:
            login(token=self.hf_token, write_permission=False)
    
    def _setup_model_paths(self):
        """Set up model and tokenizer paths based on provider"""
        if self.model_config.provider == LocalModelProvider.HUGGINGFACE:
            # Use Hugging Face model ID as is
            if not self.model_config.model_path:
                self.model_config.model_path = self.model_config.model_id
            
            if not self.model_config.tokenizer_path:
                self.model_config.tokenizer_path = self.model_config.model_id
        
        elif self.model_config.provider == LocalModelProvider.OLLAMA:
            # Ollama uses a REST API, so we'll set the endpoint
            self.api_endpoint = f"http://localhost:11434/api/generate"
            self.model_config.model_path = None
            self.model_config.tokenizer_path = None
        
        elif self.model_config.provider == LocalModelProvider.VLLM:
            # vLLM also uses a REST API
            self.api_endpoint = f"http://localhost:8000/v1/completions"
            if not self.model_config.model_path:
                self.model_config.model_path = self.model_config.model_id
        
        elif self.model_config.provider == LocalModelProvider.LLAMACPP:
            # Check for llama.cpp server endpoint or local model path
            self.api_endpoint = f"http://localhost:8080/completion"
            if not self.model_config.model_path:
                # Try to locate the GGUF file
                self.model_config.model_path = os.path.join(
                    self.cache_dir, 
                    f"{self.model_config.model_id.replace('/', '_')}.gguf"
                )
    
    def _download_model_if_needed(self) -> str:
        """Download the model if not already available locally"""
        if self.model_config.provider != LocalModelProvider.HUGGINGFACE:
            return self.model_config.model_path
        
        try:
            # Check if we already have the model cached
            local_path = os.path.join(self.cache_dir, self.model_config.model_id.replace('/', '_'))
            if os.path.exists(local_path) and os.listdir(local_path):
                logger.info(f"Using cached model at {local_path}")
                return local_path
            
            # Download model snapshot
            logger.info(f"Downloading model {self.model_config.model_id}...")
            return snapshot_download(
                repo_id=self.model_config.model_id,
                revision=self.model_config.revision,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                token=self.hf_token
            )
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise RuntimeError(f"Failed to download model {self.model_config.model_id}: {str(e)}")
    
    def load_model(self, force_reload: bool = False):
        """Load the model and tokenizer into memory"""
        if self.model is not None and not force_reload:
            logger.info("Model already loaded")
            return
        
        if self.model_config.provider == LocalModelProvider.HUGGINGFACE:
            self._load_huggingface_model()
        elif self.model_config.provider == LocalModelProvider.OLLAMA:
            logger.info("Using Ollama API - no local model loading needed")
            # Test connection
            self._test_ollama_connection()
        elif self.model_config.provider == LocalModelProvider.VLLM:
            logger.info("Using vLLM API - no local model loading needed")
            # Test connection
            self._test_vllm_connection()
        elif self.model_config.provider == LocalModelProvider.LLAMACPP:
            logger.info("Using llama.cpp API - no local model loading needed")
            # Test connection
            self._test_llamacpp_connection()
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
    
    def _load_huggingface_model(self):
        """Load a Hugging Face model and tokenizer"""
        try:
            model_path = self._download_model_if_needed()
            
            logger.info(f"Loading tokenizer from {self.model_config.tokenizer_path or model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.tokenizer_path or model_path,
                trust_remote_code=self.model_config.trust_remote_code,
                use_fast=True
            )
            
            # Configure quantization if needed
            quantization_config = None
            if self.model_config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.model_config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            device_map = "cpu" if self.model_config.cpu_only else self.model_config.device_map
            
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=self.model_config.get_torch_dtype(),
                device_map=device_map,
                quantization_config=quantization_config,
                max_memory=self.model_config.max_memory,
                offload_folder=self.model_config.offload_folder,
                low_cpu_mem_usage=True,
                use_cache=self.model_config.use_cache
            )
            
            # Create generation pipeline
            logger.info("Creating generation pipeline")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                use_cache=self.model_config.use_cache
            )
            
            # Set up token counter
            self.token_counter = TokenCounter(self.tokenizer)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama API"""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.model_config.model_id not in model_names:
                    logger.warning(f"Model {self.model_config.model_id} not found in Ollama. Available models: {model_names}")
                logger.info(f"Successfully connected to Ollama API. Available models: {model_names}")
                # Set up token counter with dummy tokenizer
                self.token_counter = TokenCounter()
            else:
                logger.error(f"Failed to connect to Ollama API: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {str(e)}")
            raise RuntimeError(f"Failed to connect to Ollama API: {str(e)}")
    
    def _test_vllm_connection(self):
        """Test connection to vLLM API"""
        import requests
        try:
            response = requests.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                logger.info(f"Successfully connected to vLLM API")
                # Set up token counter with dummy tokenizer
                self.token_counter = TokenCounter()
            else:
                logger.error(f"Failed to connect to vLLM API: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Error connecting to vLLM API: {str(e)}")
            raise RuntimeError(f"Failed to connect to vLLM API: {str(e)}")
    
    def _test_llamacpp_connection(self):
        """Test connection to llama.cpp API"""
        import requests
        try:
            response = requests.post(
                "http://localhost:8080/completion",
                json={"prompt": "Hello", "n_predict": 1}
            )
            if response.status_code == 200:
                logger.info(f"Successfully connected to llama.cpp API")
                # Set up token counter with dummy tokenizer
                self.token_counter = TokenCounter()
            else:
                logger.error(f"Failed to connect to llama.cpp API: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Error connecting to llama.cpp API: {str(e)}")
            raise RuntimeError(f"Failed to connect to llama.cpp API: {str(e)}")
    
    def _format_prompt(self, messages: List[LLMMessage]) -> str:
        """Format messages into a prompt string based on the model"""
        # Detect common model types and format appropriately
        model_id = self.model_config.model_id.lower()
        
        # Llama 2 Chat format
        if "llama-2" in model_id and "chat" in model_id:
            return self._format_llama2_chat(messages)
        # Llama 3 format
        elif "llama-3" in model_id:
            return self._format_llama3_chat(messages)
        # CodeLlama format
        elif "codellama" in model_id:
            return self._format_codellama(messages)
        # Mistral format
        elif "mistral" in model_id:
            return self._format_mistral(messages)
        # Zephyr format
        elif "zephyr" in model_id:
            return self._format_zephyr(messages)
        # Phi format
        elif "phi" in model_id:
            return self._format_phi(messages)
        # Mixtral format
        elif "mixtral" in model_id:
            return self._format_mixtral(messages)
        # Falcon format
        elif "falcon" in model_id:
            return self._format_falcon(messages)
        # Default to ChatML format
        else:
            return self._format_chatml(messages)
    
    def _format_llama2_chat(self, messages: List[LLMMessage]) -> str:
        """Format messages for Llama 2 Chat models"""
        system_prompt = "You are a helpful, respectful and honest assistant."
        formatted_messages = []
        
        # Extract system prompt if present
        for i, msg in enumerate(messages):
            if msg.role == "system":
                system_prompt = msg.content
                break
        
        # Build conversation with Llama 2 format
        conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        for i, msg in enumerate(messages):
            if msg.role == "system":
                continue
            elif msg.role == "user":
                if i > 0 and conversation and not conversation.endswith("[INST] "):
                    conversation += " [/INST] "
                if i > 0 and not conversation.endswith("[INST] "):
                    conversation += "[INST] "
                conversation += msg.content
            elif msg.role == "assistant":
                conversation += f" [/INST] {msg.content} </s>"
                if i < len(messages) - 1:
                    conversation += "<s>"
        
        # Close the final user message if needed
        if conversation and not conversation.endswith("[/INST] ") and not conversation.endswith("</s>"):
            conversation += " [/INST] "
        
        return conversation
    
    def _format_llama3_chat(self, messages: List[LLMMessage]) -> str:
        """Format messages for Llama 3 models"""
        formatted = "<|begin_of_text|>"
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        
        # Add system instruction if present
        if system_content:
            formatted += f"<|system|>\n{system_content}<|end_of_system|>\n"
        
        # Add messages
        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                formatted += f"<|user|>\n{msg.content}<|end_of_user|>\n"
            elif msg.role == "assistant":
                formatted += f"<|assistant|>\n{msg.content}<|end_of_assistant|>\n"
        
        # Add final assistant tag for generation
        if messages[-1].role != "assistant":
            formatted += "<|assistant|>\n"
        
        return formatted
    
    def _format_codellama(self, messages: List[LLMMessage]) -> str:
        """Format messages for CodeLlama models"""
        # CodeLlama uses the same format as Llama 2 Chat
        return self._format_llama2_chat(messages)
    
    def _format_mistral(self, messages: List[LLMMessage]) -> str:
        """Format messages for Mistral models"""
        formatted = ""
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        
        # Start with the system message if present
        if system_content:
            formatted += f"<s>[INST] {system_content} [/INST]</s>"
        
        # Add messages
        for i, msg in enumerate(messages):
            if msg.role == "system":
                continue
            elif msg.role == "user":
                formatted += f"<s>[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content}</s>"
        
        return formatted
    
    def _format_zephyr(self, messages: List[LLMMessage]) -> str:
        """Format messages for Zephyr models"""
        formatted = ""
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        
        # Process messages
        for i, msg in enumerate(messages):
            if msg.role == "system":
                continue
            elif msg.role == "user":
                if system_content and i == 0:
                    formatted += f"<|system|>\n{system_content}\n<|user|>\n{msg.content}\n"
                else:
                    formatted += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"<|assistant|>\n{msg.content}\n"
        
        # Add assistant marker for generation
        if messages[-1].role != "assistant":
            formatted += "<|assistant|>\n"
        
        return formatted
    
    def _format_phi(self, messages: List[LLMMessage]) -> str:
        """Format messages for Phi models"""
        formatted = ""
        
        # Process messages
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"Human: {msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n\n"
        
        # Add assistant prefix for generation
        if messages[-1].role != "assistant":
            formatted += "Assistant: "
        
        return formatted
    
    def _format_mixtral(self, messages: List[LLMMessage]) -> str:
        """Format messages for Mixtral models"""
        # Mixtral uses the same format as Mistral
        return self._format_mistral(messages)
    
    def _format_falcon(self, messages: List[LLMMessage]) -> str:
        """Format messages for Falcon models"""
        formatted = ""
        
        # Process messages
        for msg in messages:
            if msg.role == "system":
                formatted += f"{msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        
        # Add assistant prefix for generation
        if messages[-1].role != "assistant":
            formatted += "Assistant: "
        
        return formatted
    
    def _format_chatml(self, messages: List[LLMMessage]) -> str:
        """Format messages in ChatML format (default)"""
        formatted = ""
        
        # Process messages
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                formatted += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                formatted += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        
        # Add assistant marker for generation
        if messages[-1].role != "assistant":
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def _compute_response_tokens(self, response: str) -> int:
        """Compute the number of tokens in the response"""
        if self.tokenizer:
            return len(self.tokenizer.encode(response))
        else:
            # Fallback to rough estimation
            return len(response.split())
    
    def generate(self, messages: List[LLMMessage], options: Optional[LLMRequestOptions] = None) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of conversation messages
            options: Generation options
            
        Returns:
            LLM response
        """
        if options is None:
            options = LLMRequestOptions()
        
        # Load model if not already loaded
        if self.model is None and self.model_config.provider == LocalModelProvider.HUGGINGFACE:
            self.load_model()
        
        # Call the appropriate generation method based on provider
        if self.model_config.provider == LocalModelProvider.HUGGINGFACE:
            return self._generate_huggingface(messages, options)
        elif self.model_config.provider == LocalModelProvider.OLLAMA:
            return self._generate_ollama(messages, options)
        elif self.model_config.provider == LocalModelProvider.VLLM:
            return self._generate_vllm(messages, options)
        elif self.model_config.provider == LocalModelProvider.LLAMACPP:
            return self._generate_llamacpp(messages, options)
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
    
    def _generate_huggingface(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """Generate a response using Hugging Face pipeline"""
        try:
            # Format prompt based on model type
            prompt = self._format_prompt(messages)
            
            # Count prompt tokens
            prompt_tokens = self.token_counter.count_tokens(prompt)
            
            # Configure generation parameters
            generation_config = {
                "max_new_tokens": options.max_tokens or 2048,
                "temperature": options.temperature,
                "top_p": options.top_p,
                "top_k": options.top_k,
                "repetition_penalty": options.repetition_penalty,
                "do_sample": options.temperature > 0.0,
                "num_return_sequences": options.num_return_sequences,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Set up stopping criteria if needed
            stopping_criteria = None
            if options.stop_sequences:
                custom_stopping_criteria = CustomStoppingCriteria(options.stop_sequences, self.tokenizer)
                stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
                generation_config["stopping_criteria"] = stopping_criteria
            
            # Generate response
            if options.stream:
                # Set up streamer
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
                generation_config["streamer"] = streamer
                
                # Start generation in a separate thread
                thread = ThreadPoolExecutor(1)
                future = thread.submit(
                    self.pipeline, 
                    prompt, 
                    **generation_config
                )
                
                # Stream the response
                generated_text = ""
                for text in streamer:
                    generated_text += text
                
                # Wait for generation to complete
                _ = future.result()
            else:
                # Generate directly
                outputs = self.pipeline(
                    prompt,
                    **generation_config
                )
                
                # Extract the generated text
                generated_text = outputs[0]['generated_text']
                
                # Remove the prompt from the beginning of the response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
            
            # Count completion tokens
            completion_tokens = self._compute_response_tokens(generated_text)
            
            # Create usage metrics
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            # Create and return the response
            return LLMResponse(
                content=generated_text.strip(),
                usage=usage,
                model_id=self.model_config.model_id,
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def _generate_ollama(self, messages: List[LLMMessage], options: LLMRequestOptions) -> LLMResponse:
        """Generate a response using Ollama API"""
        try:
            import requests
            
            # Format prompt based on Ollama's expectations
            formatted_messages = []
            
            for msg in messages:
                if msg.role == "system":
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            # Estimate prompt tokens (approximate)
            prompt_text = "\n".join([m.content for m in messages])
            prompt_tokens = self.token_counter.count_tokens(prompt_text)
            
            # Prepare request
            payload = {
                "model": self.model_config.model_id,
                "messages": formatted_messages,
                "options": {
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "top_k": options.top_k,
                    "num_predict": options.max_tokens or 2048,
                    "repeat_penalty": options.repetition_penalty,
                    "seed": options.seed
                },
                "stream": options.stream
            }
            
            # Handle streaming
            if options.stream:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                
                full_response = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(f"Ollama API error: {chunk['error']}")
                    
                    if "message" in chunk:
                        message_content = chunk["message"]["content"]
                        full_response += message_content
                    
                    # Check if generation is done
                    if chunk.get("done", False):
                        break
                
                generated_text = full_response
            else:
                # Non-streaming request
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload
                )
                response.raise_for_status()
                
                response_data = response.json()
                if "error" in response_data:
                    raise RuntimeError(f"Ollama API error: {response_data['error']}")
                
                generated_text = response_data["message"]["content"]
            
            # Estimate completion tokens (approximate)
            completion_tokens = self.token_counter.count_tokens(generated_text)
            
            # Create usage metrics
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            # Create and return the response
            return LLMResponse(
                content=generated_text.strip(),
                usage=usage,
                model_id=self.model_config.model_id,
                finish_reason="stop"
            )
            