import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from model_manager import ModelManager, ModelCapability
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class UseCaseTemplate:
    """Template for specific use cases"""
    name: str
    description: str
    complexity: str  # Simple, Medium, Complex, Enterprise
    required_capabilities: List[ModelCapability]
    recommended_models: Dict[str, str]
    default_config: Dict[str, Any]
    architecture_patterns: List[str]
    technology_stack: List[str]
    special_requirements: List[str] = field(default_factory=list)

@dataclass
class EnterpriseConfig:
    """Enterprise-specific configuration"""
    multi_tenant: bool = False
    high_availability: bool = True
    disaster_recovery: bool = True
    compliance_requirements: List[str] = field(default_factory=list)
    security_level: str = "high"  # basic, medium, high, critical
    scalability_target: str = "1000_users"  # 100_users, 1000_users, 10000_users, enterprise
    deployment_environment: str = "cloud"  # local, cloud, hybrid, on_premise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "multi_tenant": self.multi_tenant,
            "high_availability": self.high_availability,
            "disaster_recovery": self.disaster_recovery,
            "compliance_requirements": self.compliance_requirements,
            "security_level": self.security_level,
            "scalability_target": self.scalability_target,
            "deployment_environment": self.deployment_environment
        }

class EnhancedConfigManager:
    """Enhanced configuration manager for enterprise SDLC framework"""
    
    # Use case templates for different application types
    USE_CASE_TEMPLATES = {
        "text_to_sql": UseCaseTemplate(
            name="Text-to-SQL System",
            description="AI-powered natural language to SQL query conversion",
            complexity="Enterprise",
            required_capabilities=[
                ModelCapability.SQL_GENERATION,
                ModelCapability.API_DESIGN,
                ModelCapability.MICROSERVICES,
                ModelCapability.SECURITY
            ],
            recommended_models={
                "sql_generation": "openai-gpt-4",
                "architecture": "microsoft/codebert-base", 
                "coding": "Salesforce/codet5-large",
                "documentation": "EleutherAI/gpt-neo-2.7B"
            },
            default_config={
                "max_tokens": 2048,
                "temperature": 0.3,
                "enable_caching": True,
                "api_rate_limit": 1000,
                "concurrent_users": 1000
            },
            architecture_patterns=[
                "Microservices",
                "API Gateway",
                "Event-Driven Architecture",
                "CQRS",
                "Database per Service"
            ],
            technology_stack=[
                "Python", "FastAPI", "PostgreSQL", "Redis", "RabbitMQ",
                "React", "TypeScript", "Docker", "Kubernetes", "Prometheus"
            ],
            special_requirements=[
                "Multi-database support",
                "Query optimization", 
                "Real-time processing",
                "Natural language processing",
                "SQL injection prevention"
            ]
        ),
        
        "microservices_platform": UseCaseTemplate(
            name="Microservices Platform",
            description="Enterprise microservices platform with service mesh",
            complexity="Enterprise",
            required_capabilities=[
                ModelCapability.MICROSERVICES,
                ModelCapability.API_DESIGN,
                ModelCapability.SECURITY,
                ModelCapability.ARCHITECTURE
            ],
            recommended_models={
                "architecture": "microsoft/codebert-base",
                "coding": "codeparrot/codeparrot",
                "api_design": "Salesforce/codet5-large",
                "documentation": "EleutherAI/gpt-neo-2.7B"
            },
            default_config={
                "max_tokens": 1536,
                "temperature": 0.4,
                "service_mesh": True,
                "distributed_tracing": True
            },
            architecture_patterns=[
                "Microservices",
                "Service Mesh",
                "API Gateway",
                "Circuit Breaker",
                "Saga Pattern"
            ],
            technology_stack=[
                "Python", "Go", "FastAPI", "gRPC", "Istio", "Envoy",
                "Kubernetes", "Helm", "Prometheus", "Grafana", "Jaeger"
            ]
        ),
        
        "enterprise_web_app": UseCaseTemplate(
            name="Enterprise Web Application",
            description="Full-stack enterprise web application",
            complexity="Complex",
            required_capabilities=[
                ModelCapability.GENERAL,
                ModelCapability.CODING,
                ModelCapability.SECURITY,
                ModelCapability.DATABASE_DESIGN
            ],
            recommended_models={
                "general": "EleutherAI/gpt-neo-2.7B",
                "coding": "codeparrot/codeparrot",
                "database": "microsoft/codebert-base",
                "security": "Salesforce/codet5-large"
            },
            default_config={
                "max_tokens": 1024,
                "temperature": 0.5,
                "enable_ssr": True,
                "enable_pwa": True
            },
            architecture_patterns=[
                "MVC",
                "Repository Pattern",
                "Dependency Injection",
                "Clean Architecture"
            ],
            technology_stack=[
                "Python", "Django/Flask", "PostgreSQL", "Redis", "Celery",
                "React", "TypeScript", "Docker", "AWS/Azure"
            ]
        ),
        
        "simple_crud_app": UseCaseTemplate(
            name="Simple CRUD Application",
            description="Basic create, read, update, delete application",
            complexity="Simple",
            required_capabilities=[
                ModelCapability.GENERAL,
                ModelCapability.CODING
            ],
            recommended_models={
                "general": "microsoft/DialoGPT-large",
                "coding": "distilgpt2"
            },
            default_config={
                "max_tokens": 512,
                "temperature": 0.6,
                "simple_deployment": True
            },
            architecture_patterns=["MVC", "Monolithic"],
            technology_stack=["Python", "Flask", "SQLite", "HTML", "CSS", "JavaScript"]
        )
    }
    
    def __init__(self):
        self.model_info = ModelManager.get_comprehensive_info()
        self.enterprise_config = self._load_enterprise_config()
        self.current_template = None
        self.agent_configs = {}
        self._setup_default_configs()
    
    def _load_enterprise_config(self) -> EnterpriseConfig:
        """Load enterprise configuration from environment"""
        return EnterpriseConfig(
            multi_tenant=os.getenv("MULTI_TENANT", "false").lower() == "true",
            high_availability=os.getenv("HIGH_AVAILABILITY", "true").lower() == "true", 
            disaster_recovery=os.getenv("DISASTER_RECOVERY", "true").lower() == "true",
            compliance_requirements=os.getenv("COMPLIANCE_REQUIREMENTS", "").split(","),
            security_level=os.getenv("SECURITY_LEVEL", "high"),
            scalability_target=os.getenv("SCALABILITY_TARGET", "1000_users"),
            deployment_environment=os.getenv("DEPLOYMENT_ENV", "cloud")
        )
    
    def _setup_default_configs(self):
        """Setup default configurations for all agent types"""
        model_routing = ModelManager.get_model_routing_strategy()
        
        self.agent_configs = {
            "requirements": {
                "model_name": model_routing.get(ModelCapability.GENERAL),
                "max_tokens": 1024,
                "temperature": 0.7,
                "description": "Converts business requirements to user stories"
            },
            "architecture": {
                "model_name": model_routing.get(ModelCapability.ARCHITECTURE),
                "max_tokens": 1536,
                "temperature": 0.6,
                "description": "Generates system architecture and technical specifications"
            },
            "coding": {
                "model_name": model_routing.get(ModelCapability.CODING),
                "max_tokens": 2048,
                "temperature": 0.3,
                "description": "Generates application code and microservices"
            },
            "documentation": {
                "model_name": model_routing.get(ModelCapability.DOCUMENTATION),
                "max_tokens": 1024,
                "temperature": 0.5,
                "description": "Creates comprehensive documentation"
            }
        }
    
    def load_use_case_template(self, use_case_type: str) -> bool:
        """Load configuration for a specific use case"""
        if use_case_type not in self.USE_CASE_TEMPLATES:
            logger.warning(f"Unknown use case type: {use_case_type}")
            return False
        
        self.current_template = self.USE_CASE_TEMPLATES[use_case_type]
        
        # Update agent configurations based on template
        for capability, model_name in self.current_template.recommended_models.items():
            if capability in self.agent_configs:
                # Check if recommended model is available
                if ModelManager.can_run_model(model_name):
                    self.agent_configs[capability]["model_name"] = model_name
                else:
                    # Fallback to best available model for the capability
                    cap_enum = ModelCapability(capability) if isinstance(capability, str) else capability
                    fallback_model = ModelManager.get_best_model_for_capability(cap_enum)
                    self.agent_configs[capability]["model_name"] = fallback_model
                
                # Update configuration from template defaults
                for key, value in self.current_template.default_config.items():
                    if key in ["max_tokens", "temperature"]:
                        self.agent_configs[capability][key] = value
        
        logger.info(f"Loaded configuration for use case: {use_case_type}")
        return True
    
    def get_config_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        config = self.agent_configs.get(agent_type, self.agent_configs["requirements"]).copy()
        
        # Add enterprise-specific configurations with serializable data
        config.update({
            "enterprise_config": self.enterprise_config.to_dict(),
            "use_case_template": self.current_template.name if self.current_template else None,
            "security_level": self.enterprise_config.security_level,
            "scalability_target": self.enterprise_config.scalability_target
        })
        
        return config
    
    def get_use_case_analysis(self, business_requirements: str) -> Dict[str, Any]:
        """Analyze business requirements to suggest appropriate use case template"""
        requirements_lower = business_requirements.lower()
        
        # Keyword mapping for automatic use case detection
        use_case_keywords = {
            "text_to_sql": ["sql", "query", "database", "natural language", "text to sql", "data analysis"],
            "microservices_platform": ["microservices", "api gateway", "distributed", "scalable", "enterprise platform"],
            "enterprise_web_app": ["web application", "enterprise", "user management", "dashboard", "portal"],
            "simple_crud_app": ["crud", "simple", "basic", "manage", "create", "update", "delete"]
        }
        
        scores = {}
        for use_case, keywords in use_case_keywords.items():
            score = sum(1 for keyword in keywords if keyword in requirements_lower)
            scores[use_case] = score
        
        # Get best match
        best_match = max(scores.items(), key=lambda x: x[1])
        suggested_use_case = best_match[0] if best_match[1] > 0 else "enterprise_web_app"
        
        return {
            "suggested_use_case": suggested_use_case,
            "confidence_scores": scores,
            "template_info": self.USE_CASE_TEMPLATES[suggested_use_case],
            "analysis": self._analyze_complexity(business_requirements)
        }
    
    def _analyze_complexity(self, requirements: str) -> Dict[str, Any]:
        """Analyze requirement complexity"""
        complexity_indicators = {
            "simple": ["basic", "simple", "crud", "manage", "list"],
            "medium": ["integration", "api", "authentication", "users", "search"],
            "complex": ["microservices", "scalable", "performance", "security", "multi-tenant"],
            "enterprise": ["compliance", "audit", "sso", "enterprise", "distributed", "monitoring"]
        }
        
        requirements_lower = requirements.lower()
        complexity_scores = {}
        
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in requirements_lower)
            complexity_scores[level] = score
        
        estimated_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "estimated_complexity": estimated_complexity,
            "complexity_scores": complexity_scores,
            "estimated_timeline": self._estimate_timeline(estimated_complexity),
            "team_size_recommendation": self._recommend_team_size(estimated_complexity)
        }
    
    def _estimate_timeline(self, complexity: str) -> str:
        """Estimate project timeline based on complexity"""
        timelines = {
            "simple": "2-4 weeks",
            "medium": "1-3 months", 
            "complex": "3-6 months",
            "enterprise": "6-12 months"
        }
        return timelines.get(complexity, "3-6 months")
    
    def _recommend_team_size(self, complexity: str) -> str:
        """Recommend team size based on complexity"""
        team_sizes = {
            "simple": "1-2 developers",
            "medium": "2-4 developers",
            "complex": "4-8 developers", 
            "enterprise": "8-15 developers"
        }
        return team_sizes.get(complexity, "4-8 developers")
    
    def get_system_summary(self) -> str:
        """Get a summary of the system configuration"""
        summary = f"""
System Configuration:
- CUDA Available: {self.model_info['cuda_available']}
- GPU Memory: {self.model_info['gpu_memory']:.1f} GB
- Available RAM: {self.model_info['available_ram']:.1f} GB
- Runnable Models: {self.model_info['runnable_models']}/{self.model_info['total_models']}

Enterprise Configuration:
- Security Level: {self.enterprise_config.security_level}
- High Availability: {self.enterprise_config.high_availability}
- Multi-Tenant: {self.enterprise_config.multi_tenant}
- Deployment Environment: {self.enterprise_config.deployment_environment}

Current Template: {self.current_template.name if self.current_template else 'None'}
"""
        
        if self.current_template:
            summary += f"""
Template Details:
- Complexity: {self.current_template.complexity}
- Architecture: {', '.join(self.current_template.architecture_patterns[:3])}
- Tech Stack: {', '.join(self.current_template.technology_stack[:5])}
"""
        
        return summary

# Global configuration instance
config = EnhancedConfigManager()
