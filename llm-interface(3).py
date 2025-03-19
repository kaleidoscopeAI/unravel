#!/usr/bin/env python3
"""
Unravel AI - LLM Integration Module
Interfaces with Language Models to enhance code analysis, specification generation,
and software reconstruction capabilities.
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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local" 

class LLMModel(Enum):
    """Supported LLM models"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_32K = "gpt-4-32k"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"

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
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    seed: Optional[int] = None

@dataclass
class LLMUsage:
    """Tracks token usage"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class TokenCounter:
    """Utility for counting tokens in LLM requests"""
    
    def __init__(self, model: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in a list of messages"""
        token_count = 0
        
        for message in messages:
            # Add message metadata tokens (depends on the model)
            token_count += 4  # Approximate overhead per message
            
            # Add content tokens
            token_count += self.count_tokens(message.content)
        
        # Add extra tokens for the message format
        token_count += 2  # Approximate overhead for the entire message list
        
        return token_count
    
    def split_into_chunks(self, text: str, max_tokens: int = 8000, 
                           overlap: int = 500) -> List[str]:
        """Split text into overlapping chunks that fit token limits"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for line in lines:
            line_token_count = self.count_tokens(line)
            
            if current_token_count + line_token_count > max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Calculate overlap
                overlap_start = max(0, len(current_chunk) - self._calculate_overlap_lines(current_chunk, overlap))
                current_chunk = current_chunk[overlap_start:]
                current_token_count = self.count_tokens('\n'.join(current_chunk))
            
            current_chunk.append(line)
            current_token_count += line_token_count
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _calculate_overlap_lines(self, lines: List[str], target_tokens: int) -> int:
        """Calculate how many lines to include in the overlap"""
        total_lines = len(lines)
        token_count = 0
        lines_needed = 0
        
        for i in range(total_lines - 1, -1, -1):
            line_tokens = self.count_tokens(lines[i])
            if token_count + line_tokens > target_tokens:
                break
            
            token_count += line_tokens
            lines_needed += 1
            
            if lines_needed >= total_lines // 2:
                break
                
        return lines_needed

class LLMInterface:
    """Main interface for communicating with Language Models"""
    
    def __init__(self, 
                provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
                model: Union[str, LLMModel] = LLMModel.GPT4_TURBO,
                api_key: Optional[str] = None,
                api_endpoint: Optional[str] = None,
                timeout: int = 120,
                cache_dir: Optional[str] = None):
        """
        Initialize the LLM interface
        
        Args:
            provider: LLM provider (OpenAI, Anthropic, etc.)
            model: LLM model to use
            api_key: API key for the LLM provider
            api_endpoint: Custom API endpoint URL
            timeout: Request timeout in seconds
            cache_dir: Directory to cache responses
        """
        # Normalize provider and model
        self.provider = provider if isinstance(provider, LLMProvider) else LLMProvider(provider)
        self.model = model if isinstance(model, LLMModel) else model
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or self._get_api_key_from_env()
        
        # Set API endpoint based on provider
        self.api_endpoint = api_endpoint or self._get_default_endpoint()
        
        # Set request timeout
        self.timeout = timeout
        
        # Set up caching if configured
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize token counter
        self.token_counter = TokenCounter(str(self.model) if isinstance(self.model, LLMModel) else self.model)
        
        # Initialize session in async methods
        self._session = None
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variables based on provider"""
        env_var = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.AZURE: "AZURE_OPENAI_API_KEY",
            LLMProvider.LOCAL: "LOCAL_LLM_API_KEY"
        }.get(self.provider, "LLM_API_KEY")
        
        api_key = os.environ.get(env_var)
        if not api_key:
            logger.warning(f"API key not found in environment variable {env_var}")
        
        return api_key
    
    def _get_default_endpoint(self) -> str:
        """Get default API endpoint based on provider"""
        return {
            LLMProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
            LLMProvider.AZURE: "https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2023-05-15",
            LLMProvider.LOCAL: "http://localhost:8000/v1/chat/completions"
        }.get(self.provider, "https://api.openai.com/v1/chat/completions")
    
    def _format_request(self, messages: List[LLMMessage], options: LLMRequestOptions) -> Dict[str, Any]:
        """Format the request payload based on the provider"""
        if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.AZURE:
            return {
                "model": self.model.value if isinstance(self.model, LLMModel) else self.model,
                "messages": [msg.to_dict() for msg in messages],
                "temperature": options.temperature,
                "top_p": options.top_p,
                "max_tokens": options.max_tokens,
                "stop": options.stop_sequences if options.stop_sequences else None,
                "stream": options.stream,
                "seed": options.seed
            }
        
        elif self.provider == LLMProvider.ANTHROPIC:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_content = None
            
            for msg in messages:
                if msg.role == "system":
                    system_content = msg.content
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.content})
            
            payload = {
                "model": self.model.value if isinstance(self.model, LLMModel) else self.model,
                "messages": anthropic_messages,
                "temperature": options.temperature,
                "top_p": options.top_p,
                "max_tokens": options.max_tokens or 4096,
                "stop_sequences": options.stop_sequences,
                "stream": options.stream
            }
            
            if system_content:
                payload["system"] = system_content
            
            return payload
        
        elif self.provider == LLMProvider.LOCAL:
            # Most local APIs use OpenAI-compatible format
            return {
                "model": self.model.value if isinstance(self.model, LLMModel) else self.model,
                "messages": [msg.to_dict() for msg in messages],
                "temperature": options.temperature,
                "top_p": options.top_p,
                "max_tokens": options.max_tokens,
                "stop": options.stop_sequences if options.stop_sequences else None,
                "stream": options.stream
            }
        
        raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> Tuple[str, LLMUsage]:
        """Parse response from LLM API"""
        if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.