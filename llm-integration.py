"""
LLM Integration for Kaleidoscope AI
"""

import os
import json
import time
import logging
import hashlib
import asyncio
import aiohttp
import tiktoken
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import config

logger = logging.getLogger(__name__)

class TokenCounter:
    """Utility for counting tokens in prompts"""
    
    def __init__(self, model: str = None):
        """Initialize the token counter"""
        self.model = model or config.get("LLM_MODEL", "gpt-4")
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            logger.warning(f"Model '{self.model}' not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in chat messages"""
        total_tokens = 0
        
        for message in messages:
            # Count message metadata tokens
            total_tokens += 4  # approximation for message formatting
            
            # Count content tokens
            if "content" in message and message["content"]:
                total_tokens += self.count_tokens(message["content"])
        
        # Add tokens for the assistant's reply format
        total_tokens += 2  # approximation for reply formatting
        
        return total_tokens

class LLMChunkManager:
    """Manages chunking of large files for LLM processing"""
    
    def __init__(self, max_tokens: int = 8000, overlap: int = 500):
        """Initialize the chunk manager"""
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.token_counter = TokenCounter()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # Split text into lines
        lines = text.split('\n')
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for line in lines:
            line_token_count = self.token_counter.count_tokens(line)
            
            # If adding this line would exceed the limit, finalize the current chunk
            if current_token_count + line_token_count > self.max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.calculate_overlap_lines(current_chunk))
                current_chunk = current_chunk[overlap_start:]
                current_token_count = self.token_counter.count_tokens('\n'.join(current_chunk))
            
            # Add the current line
            current_chunk.append(line)
            current_token_count += line_token_count
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def calculate_overlap_lines(self, lines: List[str]) -> int:
        """Calculate how many lines to include in the overlap"""
        target_tokens = self.overlap
        total_lines = len(lines)
        token_count = 0
        lines_needed = 0
        
        # Count backward from the end to find overlap lines
        for i in range(total_lines - 1, -1, -1):
            line_tokens = self.token_counter.count_tokens(lines[i])
            if token_count + line_tokens > target_tokens:
                break
            
            token_count += line_tokens
            lines_needed += 1
            
            if lines_needed >= total_lines // 2:
                # Don't use more than half the chunk for overlap
                break
        
        return lines_needed

class LLMCache:
    """Caches LLM responses to avoid redundant API calls"""
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        """Initialize the cache"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate a cache key"""
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the prompt and parameters
        combined = prompt + param_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Get a cached response if available"""
        cache_key = self._get_cache_key(prompt, params)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
                return None
        
        return None
    
    def cache_response(self, prompt: str, params: Dict[str, Any], response: str) -> bool:
        """Cache a response"""
        cache_key = self._get_cache_key(prompt, params)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(response)
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {str(e)}")
            return False

class LLMIntegration:
    """LLM integration for code analysis"""
    
    def __init__(self):
        """Initialize LLM integration"""
        self.api_key = config.get("LLM_API_KEY", "")
        self.model = config.get("LLM_MODEL", "gpt-4")
        self.endpoint = config.get("LLM_ENDPOINT", "http://localhost:8000/v1")
        self.cache = LLMCache()
        self.chunk_manager = LLMChunkManager()
        self.token_counter = TokenCounter(self.model)
        
    async def analyze_code(
        self, 
        code: str, 
        task: str = "decompilation", 
        language: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Analyze code using LLM"""
        logger.info(f"Analyzing code for task: {task}")
        
        # Check if code is large and requires chunking
        code_tokens = self.token_counter.count_tokens(code)
        if code_tokens > self.chunk_manager.max_tokens // 2:
            logger.info(f"Code is large ({code_tokens} tokens), splitting into chunks")
            return await self._analyze_chunked_code(code, task, language, use_cache, **kwargs)
        
        # Create prompt
        prompt = self._create_prompt(task, code, language)
        
        # Check cache
        if use_cache:
            params = {
                "model": self.model,
                "task": task,
                "language": language,
                **kwargs
            }
            
            cached_response = self.cache.get_cached_response(prompt, params)
            if cached_response:
                logger.info("Using cached LLM response")
                return cached_response
        
        # Call LLM API
        response = await self._call_llm_api(prompt, **kwargs)
        
        # Cache response
        if use_cache:
            self.cache.cache_response(prompt, params, response)
        
        return response
    
    async def _analyze_chunked_code(
        self, 
        code: str, 
        task: str, 
        language: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Analyze large code by splitting it into chunks"""
        chunks = self.chunk_manager.split_text_into_chunks(code)
        logger.info(f"Split code into {len(chunks)} chunks")
        
        chunk_responses = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create chunk prompt
            chunk_prompt = self._create_prompt(
                task, 
                f"CHUNK {i+1} OF {len(chunks)}\n\n{chunk}", 
                language,
                is_chunk=True
            )
            
            # Check cache
            if use_cache:
                params = {
                    "model": self.model,
                    "task": task,
                    "language": language,
                    "chunk": i+1,
                    "total_chunks": len(chunks),
                    **kwargs
                }
                
                cached_response = self.cache.get_cached_response(chunk_prompt, params)
                if cached_response:
                    logger.info(f"Using cached LLM response for chunk {i+1}")
                    chunk_responses.append(cached_response)
                    continue
            
            # Call LLM API
            response = await self._call_llm_api(chunk_prompt, **kwargs)
            chunk_responses.append(response)
            
            # Cache response
            if use_cache:
                self.cache.cache_response(chunk_prompt, params, response)
        
        # Create summary prompt to combine chunks
        summary_prompt = f"""You have analyzed a large codebase in chunks. Below are your analyses for each chunk:

{chr(10).join([f"CHUNK {i+1}:\n{response}\n" for i, response in enumerate(chunk_responses)])}

Please provide a comprehensive summary that combines the information from all chunks into a coherent analysis. Focus on the most important findings, patterns, and insights across the entire codebase.
"""
        
        # Get combined analysis
        combined_analysis = await self._call_llm_api(summary_prompt, **kwargs)
        
        return combined_analysis
    
    def _create_prompt(
        self, 
        task: str, 
        code: str, 
        language: Optional[str] = None,
        is_chunk: bool = False
    ) -> str:
        """Create prompt for code analysis"""
        prompts = {
            "decompilation": """
You are an expert software reverse engineer. Analyze the following code thoroughly:

{code}

First, identify the purpose and functionality of this code.
Then, extract the core logic and algorithms.
Finally, provide a clean, restructured version with meaningful variable and function names.

Your response should include:
1. An overview of the code's purpose
2. Key structures and functions
3. A clean, readable version of the code
""",
            "structure_analysis": """
You are an expert software architect. Analyze the following code and extract its structure:

{code}

Your response should include:
1. High-level architecture overview
2. Main classes/modules and their responsibilities
3. Key dependencies and relationships
4. Data flow and control flow
5. Any design patterns identified

Format the output as a structured analysis that could be used to recreate this software from scratch.
""",
            "specification": """
Based on the analysis of the software, create a comprehensive specification document.

{code}

The specification should:
1. Define the purpose and scope of the software
2. List all functional requirements
3. Describe data structures and relationships
4. Detail APIs and interfaces
5. Outline key algorithms and logic

This specification will be used as the foundation for reconstructing the software, so it should be complete and precise.
""",
            "reconstruction": """
You are tasked with reconstructing software based on the following specification:

{code}

Create clean, well-structured code that implements this specification. Your code should:
1. Follow modern best practices