import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from typing import Optional, List, Dict, Any
import asyncio
import threading
import aiohttp
import json
from abc import ABC, abstractmethod
from model_manager import ModelSpec, ModelManager

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class HuggingFaceProvider(LLMProvider):
    """Hugging Face local model provider"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading Hugging Face model {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                trust_remote_code=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate text asynchronously"""
        def _generate():
            try:
                # Generate text
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                    length_penalty=kwargs.get('length_penalty', 1.0)
                )
                
                # Extract generated text (remove input prompt)
                generated_text = outputs[0]["generated_text"]
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate)
    
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def _get_api_key(self) -> Optional[str]:
        import os
        return os.getenv("OPENAI_API_KEY")
    
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        return self.api_key is not None

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def _get_api_key(self) -> Optional[str]:
        import os
        return os.getenv("ANTHROPIC_API_KEY")
    
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using Anthropic API"""
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        return self.api_key is not None

class EnhancedLLMClient:
    """Enhanced LLM client with multiple providers and fallback mechanisms"""
    
    def __init__(self, primary_model: str, fallback_models: Optional[List[str]] = None):
        self.primary_model = primary_model
        self.fallback_models = fallback_models or ModelManager.get_fallback_chain(primary_model)
        self.providers = {}
        self.current_provider = None
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        all_models = [self.primary_model] + self.fallback_models
        
        for model_name in all_models:
            try:
                if model_name in ModelManager.MODEL_CATALOG:
                    spec = ModelManager.MODEL_CATALOG[model_name]
                    
                    if spec.provider == "openai":
                        provider = OpenAIProvider(model_name)
                    elif spec.provider == "anthropic":
                        provider = AnthropicProvider(model_name)
                    else:  # Default to Hugging Face
                        provider = HuggingFaceProvider(model_name)
                    
                    if provider.is_available():
                        self.providers[model_name] = provider
                        if self.current_provider is None:
                            self.current_provider = model_name
                            logger.info(f"Primary provider set to: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize provider for {model_name}: {str(e)}")
        
        if not self.providers:
            raise RuntimeError("No LLM providers could be initialized")
    
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate text with automatic fallback"""
        # Try current provider first
        if self.current_provider and self.current_provider in self.providers:
            try:
                return await self.providers[self.current_provider].generate(
                    prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                logger.warning(f"Primary provider {self.current_provider} failed: {str(e)}")
        
        # Try fallback providers
        for model_name, provider in self.providers.items():
            if model_name != self.current_provider:
                try:
                    logger.info(f"Trying fallback provider: {model_name}")
                    result = await provider.generate(prompt, max_tokens, temperature, **kwargs)
                    # Update current provider to successful one
                    self.current_provider = model_name
                    return result
                except Exception as e:
                    logger.warning(f"Fallback provider {model_name} failed: {str(e)}")
                    continue
        
        raise RuntimeError("All LLM providers failed")
    
    async def generate_with_retries(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Generate with retry logic"""
        for attempt in range(max_retries):
            try:
                return await self.generate(prompt, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            "primary_model": self.primary_model,
            "current_provider": self.current_provider,
            "available_providers": list(self.providers.keys()),
            "fallback_chain": self.fallback_models
        }
    
    def switch_provider(self, model_name: str) -> bool:
        """Manually switch to a specific provider"""
        if model_name in self.providers:
            self.current_provider = model_name
            logger.info(f"Switched to provider: {model_name}")
            return True
        return False

# Backward compatibility
LLMClient = EnhancedLLMClient
