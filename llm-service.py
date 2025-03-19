import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM inference using Mistral-7B-Instruct"""
    
    def __init__(self):
        """Initialize the LLM service with the local Mistral model"""
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.max_length = 4096
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path("/app/models/mistral7b_onnx")
        self.tokenizer_path = Path("/app/models/mistral7b_tokenizer")
        
        # Create a lock for thread-safe initialization
        self._init_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the model and tokenizer"""
        if self.initialized:
            return
        
        async with self._init_lock:
            if self.initialized:  # Check again in case another thread initialized while waiting
                return
                
            logger.info(f"Initializing LLM service on device: {self.device}")
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    padding_side="left",
                    truncation_side="left",
                )
                
                # Load model with optimized ONNX runtime
                self.model = ORTModelForCausalLM.from_pretrained(
                    self.model_path,
                    provider="CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider",
                )
                
                self.initialized = True
                logger.info("LLM service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM service: {str(e)}")
                raise
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the Mistral-7B-Instruct model"""
        return f"<s>[INST] {prompt} [/INST]"
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for the Mistral-7B-Instruct model"""
        prompt = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                if i == 0:
                    prompt += f"<s>[INST] {content} [/INST]"
                else:
                    prompt += f"</s><s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} "
            else:  # system message
                if i == 0:
                    prompt += f"<s>[INST] {content} "
                else:
                    prompt += f"</s><s>[INST] {content} "
        
        return prompt
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion for a single prompt"""
        await self.initialize()
        
        formatted_prompt = self._format_prompt(prompt)
        
        # Set default parameters
        max_new_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        response_text = generated_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return response_text.strip()
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion from a list of messages"""
        await self.initialize()
        
        formatted_prompt = self._format_chat_prompt(messages)
        
        # Set default parameters
        max_new_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        response_text = generated_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        # Format response like OpenAI's API
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion"""
        await self.initialize()
        
        formatted_prompt = self._format_chat_prompt(messages)
        
        # Set default parameters
        max_new_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate with streaming
        streamer_kwargs = {"skip_special_tokens": True, "skip_prompt": True}
        
        generated_texts = []
        
        # We'll simulate streaming by generating the full response and then yielding it token by token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Get the generated tokens (excluding prompt)
            generated_tokens = outputs[0][input_length:]
            
            # Decode token by token to simulate streaming
            last_idx = 0
            for i in range(1, len(generated_tokens) + 1):
                # Decode a small chunk of tokens
                chunk = self.tokenizer.decode(generated_tokens[last_idx:i], skip_special_tokens=True)
                if chunk.strip():
                    yield chunk
                    last_idx = i
                
                # Add a small delay to simulate processing time
                await asyncio.sleep(0.01)

# Instantiate a single service to be used application-wide
llm_service = LLMService()
