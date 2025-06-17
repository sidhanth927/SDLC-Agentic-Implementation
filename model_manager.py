import torch
import psutil
import logging
from typing import Dict, List, Optional, Tuple
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelCapability(Enum):
    """Model capability types"""
    GENERAL = "general"
    CODING = "coding"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    SQL_GENERATION = "sql_generation"
    API_DESIGN = "api_design"
    SECURITY = "security"
    TESTING = "testing"
    MICROSERVICES = "microservices"
    DATABASE_DESIGN = "database_design"

@dataclass
class ModelSpec:
    """Specification for a model"""
    name: str
    capabilities: List[ModelCapability]
    min_gpu_memory: float  # GB
    min_ram: float  # GB
    max_tokens: int
    cost_per_token: float = 0.0
    provider: str = "huggingface"
    api_endpoint: Optional[str] = None
    specialized_prompts: Dict[str, str] = None

class ModelManager:
    """Enhanced model manager for enterprise applications"""
    
    # Enhanced model catalog with specialized capabilities
    MODEL_CATALOG = {
        # General Purpose Models
        "EleutherAI/gpt-neo-2.7B": ModelSpec(
            name="EleutherAI/gpt-neo-2.7B",
            capabilities=[ModelCapability.GENERAL, ModelCapability.DOCUMENTATION],
            min_gpu_memory=8.0,
            min_ram=16.0,
            max_tokens=2048
        ),
        "microsoft/DialoGPT-large": ModelSpec(
            name="microsoft/DialoGPT-large",
            capabilities=[ModelCapability.GENERAL],
            min_gpu_memory=4.0,
            min_ram=8.0,
            max_tokens=1024
        ),
        
        # Coding Specialized Models
        "codeparrot/codeparrot": ModelSpec(
            name="codeparrot/codeparrot",
            capabilities=[ModelCapability.CODING, ModelCapability.API_DESIGN],
            min_gpu_memory=6.0,
            min_ram=12.0,
            max_tokens=2048
        ),
        "Salesforce/codet5-large": ModelSpec(
            name="Salesforce/codet5-large",
            capabilities=[ModelCapability.CODING, ModelCapability.TESTING],
            min_gpu_memory=8.0,
            min_ram=16.0,
            max_tokens=2048
        ),
        
        # Architecture & Design Models
        "microsoft/codebert-base": ModelSpec(
            name="microsoft/codebert-base",
            capabilities=[ModelCapability.ARCHITECTURE, ModelCapability.MICROSERVICES],
            min_gpu_memory=4.0,
            min_ram=8.0,
            max_tokens=1024
        ),
        
        # Lightweight fallback models
        "distilgpt2": ModelSpec(
            name="distilgpt2",
            capabilities=[ModelCapability.GENERAL, ModelCapability.DOCUMENTATION],
            min_gpu_memory=0.0,
            min_ram=2.0,
            max_tokens=512
        ),
        "gpt2": ModelSpec(
            name="gpt2",
            capabilities=[ModelCapability.GENERAL],
            min_gpu_memory=0.0,
            min_ram=2.0,
            max_tokens=512
        ),
        
        # External API Models (when available)
        "openai-gpt-4": ModelSpec(
            name="gpt-4",
            capabilities=[cap for cap in ModelCapability],
            min_gpu_memory=0.0,
            min_ram=1.0,
            max_tokens=8192,
            cost_per_token=0.00003,
            provider="openai",
            api_endpoint="https://api.openai.com/v1/chat/completions"
        ),
        "anthropic-claude": ModelSpec(
            name="claude-3-sonnet",
            capabilities=[cap for cap in ModelCapability],
            min_gpu_memory=0.0,
            min_ram=1.0,
            max_tokens=4096,
            cost_per_token=0.000015,
            provider="anthropic",
            api_endpoint="https://api.anthropic.com/v1/messages"
        )
    }
    
    @classmethod
    def get_system_info(cls) -> Dict:
        """Enhanced system information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory": 0.0,
            "gpu_memory_per_device": [],
            "total_ram": psutil.virtual_memory().total / (1024**3),
            "available_ram": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "platform": sys.platform
        }
        
        if info["cuda_available"]:
            try:
                for i in range(info["gpu_count"]):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    info["gpu_memory_per_device"].append(gpu_memory)
                    info["gpu_memory"] = max(info["gpu_memory"], gpu_memory)
            except:
                info["gpu_memory"] = 0.0
        
        return info
    
    @classmethod
    def can_run_model(cls, model_name: str) -> bool:
        """Check if system can run the specified model"""
        if model_name not in cls.MODEL_CATALOG:
            return True  # Assume compatibility for unknown models
        
        model_spec = cls.MODEL_CATALOG[model_name]
        system_info = cls.get_system_info()
        
        # External API models are always "runnable" if we have internet
        if model_spec.provider in ["openai", "anthropic", "google"]:
            return True
        
        # Check RAM requirement
        if model_spec.min_ram > system_info["available_ram"]:
            return False
        
        # Check GPU requirement
        if model_spec.min_gpu_memory > 0:
            if not system_info["cuda_available"]:
                return False
            if model_spec.min_gpu_memory > system_info["gpu_memory"]:
                return False
        
        return True
    
    @classmethod
    def get_best_model_for_capability(cls, capability: ModelCapability, prefer_local: bool = True) -> str:
        """Get the best available model for a specific capability with improved API key handling"""
        # Filter models by capability
        capable_models = []
        for model_name, spec in cls.MODEL_CATALOG.items():
            if capability in spec.capabilities and cls.can_run_model(model_name):
                capable_models.append((model_name, spec))
        
        if not capable_models:
            # Fallback to general models
            if capability != ModelCapability.GENERAL:
                return cls.get_best_model_for_capability(ModelCapability.GENERAL, prefer_local)
            else:
                return "distilgpt2"  # Final fallback
        
        # Sort by preference: local first (if preferred), then by capability
        def model_score(item):
            model_name, spec = item
            score = 0
            
            # Check if API keys are available for external models
            if spec.provider in ["openai", "anthropic"]:
                import os
                api_key = None
                if spec.provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                elif spec.provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                
                # If no API key, heavily penalize external models
                if not api_key:
                    score -= 100000  # Increased penalty to ensure local models are preferred
                    logger.info(f"No API key for {spec.provider}, preferring local models")
                else:
                    score += 500  # Prefer API models if keys are available
            
            # Strongly prefer local models when no API keys or when explicitly requested
            if spec.provider == "huggingface":
                score += 5000  # Very strong preference for local models
            
            # Prefer more capable models but not at the expense of availability
            score += len(spec.capabilities) * 50  # Reduced weight
            
            # For local models, prefer smaller token limits to avoid memory issues
            if spec.provider == "huggingface":
                score += min(spec.max_tokens, 1024) / 100
            else:
                score += spec.max_tokens / 100
            
            # Penalize resource requirements for local models
            if spec.provider == "huggingface":
                score -= spec.min_gpu_memory * 10
                score -= spec.min_ram * 5
            
            return score
        
        capable_models.sort(key=model_score, reverse=True)
        best_model = capable_models[0][0]
        
        logger.info(f"Selected {best_model} for {capability.value} capability")
        return best_model
    
    @classmethod
    def get_model_routing_strategy(cls) -> Dict[ModelCapability, str]:
        """Get optimal model routing for different capabilities"""
        routing = {}
        
        for capability in ModelCapability:
            routing[capability] = cls.get_best_model_for_capability(capability)
        
        return routing
    
    @classmethod
    def get_fallback_chain(cls, primary_model: str) -> List[str]:
        """Get fallback model chain for reliability"""
        if primary_model not in cls.MODEL_CATALOG:
            return ["gpt2"]
        
        primary_spec = cls.MODEL_CATALOG[primary_model]
        fallbacks = []
        
        # Find models with similar capabilities but lower requirements
        for model_name, spec in cls.MODEL_CATALOG.items():
            if (model_name != primary_model and 
                any(cap in spec.capabilities for cap in primary_spec.capabilities) and
                spec.min_gpu_memory <= primary_spec.min_gpu_memory and
                spec.min_ram <= primary_spec.min_ram and
                cls.can_run_model(model_name)):
                fallbacks.append(model_name)
        
        # Sort by resource requirements (ascending)
        fallbacks.sort(key=lambda x: (cls.MODEL_CATALOG[x].min_gpu_memory, 
                                    cls.MODEL_CATALOG[x].min_ram))
        
        # Always include basic fallbacks
        fallbacks.extend(["distilgpt2", "gpt2"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fallbacks = []
        for model in fallbacks:
            if model not in seen:
                seen.add(model)
                unique_fallbacks.append(model)
        
        return unique_fallbacks[:5]  # Limit to 5 fallbacks
    
    @classmethod
    def estimate_cost(cls, model_name: str, token_count: int) -> float:
        """Estimate cost for using a model"""
        if model_name not in cls.MODEL_CATALOG:
            return 0.0
        
        spec = cls.MODEL_CATALOG[model_name]
        return spec.cost_per_token * token_count
    
    @classmethod
    def get_enterprise_model_recommendations(cls, use_case_type: str) -> Dict[str, str]:
        """Get model recommendations for enterprise use cases"""
        recommendations = {
            "text_to_sql": {
                "primary": cls.get_best_model_for_capability(ModelCapability.SQL_GENERATION),
                "architecture": cls.get_best_model_for_capability(ModelCapability.ARCHITECTURE),
                "api_design": cls.get_best_model_for_capability(ModelCapability.API_DESIGN),
                "documentation": cls.get_best_model_for_capability(ModelCapability.DOCUMENTATION)
            },
            "microservices": {
                "architecture": cls.get_best_model_for_capability(ModelCapability.MICROSERVICES),
                "coding": cls.get_best_model_for_capability(ModelCapability.CODING),
                "security": cls.get_best_model_for_capability(ModelCapability.SECURITY),
                "testing": cls.get_best_model_for_capability(ModelCapability.TESTING)
            },
            "enterprise_app": {
                "general": cls.get_best_model_for_capability(ModelCapability.GENERAL),
                "coding": cls.get_best_model_for_capability(ModelCapability.CODING),
                "architecture": cls.get_best_model_for_capability(ModelCapability.ARCHITECTURE),
                "database": cls.get_best_model_for_capability(ModelCapability.DATABASE_DESIGN)
            }
        }
        
        return recommendations.get(use_case_type, recommendations["enterprise_app"])
    
    @classmethod
    def get_comprehensive_info(cls) -> Dict:
        """Get comprehensive model and system information"""
        system_info = cls.get_system_info()
        model_routing = cls.get_model_routing_strategy()
        
        available_models = {}
        for capability in ModelCapability:
            available_models[capability.value] = [
                name for name, spec in cls.MODEL_CATALOG.items()
                if capability in spec.capabilities and cls.can_run_model(name)
            ]
        
        return {
            **system_info,
            "model_routing": {cap.value: model for cap, model in model_routing.items()},
            "available_models": available_models,
            "total_models": len(cls.MODEL_CATALOG),
            "runnable_models": len([m for m in cls.MODEL_CATALOG.keys() if cls.can_run_model(m)])
        }
