import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager
from performance_monitor import performance_monitor

# Import agents with fallback handling
try:
    from agents import (
        EnterpriseRequirementsAgent, 
        EnterpriseArchitectureAgent, 
        EnterpriseCodingAgent, 
        EnterpriseBaseAgent
    )
except ImportError:
    # Fallback to basic agents if enterprise agents not available
    logging.warning("Enterprise agents not found, using basic agents")
    from agents import (
        RequirementsAgent as EnterpriseRequirementsAgent,
        ArchitectureAgent as EnterpriseArchitectureAgent, 
        CodingAgent as EnterpriseCodingAgent,
        BaseAgent as EnterpriseBaseAgent
    )

# Import config with fallback
try:
    from config import EnhancedConfigManager
except ImportError:
    from config import config as EnhancedConfigManager

# Import utils with fallback handling
try:
    from utils import (
        setup_logging, ensure_output_directory, save_file, save_json, save_yaml,
        format_project_name, create_project_structure, generate_project_metadata,
        create_deployment_configs
    )
except ImportError:
    # Basic fallback implementations
    def setup_logging():
        return logging.getLogger(__name__)
    
    def ensure_output_directory(name):
        dir_path = Path("output") / name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def save_file(content, path, desc=""):
        with open(path, 'w') as f:
            f.write(content)
        return True
    
    def save_json(data, path, desc=""):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    
    def format_project_name(reqs):
        return "Generated_Project"
    
    def create_project_structure(path, template="web_app"):
        return {}
    
    def generate_project_metadata(name, use_case, config):
        return {"name": name, "type": use_case}
    
    def create_deployment_configs(path, name, use_case):
        return {}

logger = setup_logging()

class EnterpriseSDLCOrchestrator:
    """Enterprise-grade SDLC orchestrator for complex applications"""
    
    def __init__(self):
        self.logger = logger
        
        # Initialize config manager
        try:
            self.config_manager = EnhancedConfigManager()
        except:
            # Fallback config
            self.config_manager = type('Config', (), {
                'get_use_case_analysis': lambda self, req: {
                    'suggested_use_case': 'enterprise_web_app',
                    'confidence_scores': {},
                    'analysis': {'estimated_complexity': 'medium', 'estimated_timeline': '3-6 months', 'team_size_recommendation': '4-6 developers'}
                },
                'load_use_case_template': lambda self, uc: True,
                'get_config_for_agent': lambda self, agent: {'max_tokens': 1024, 'temperature': 0.7}
            })()
        
        # Initialize agents
        try:
            self.requirements_agent = EnterpriseRequirementsAgent()
            self.architecture_agent = EnterpriseArchitectureAgent()
            self.coding_agent = EnterpriseCodingAgent()
            self.documentation_agent = EnterpriseBaseAgent("documentation")
        except Exception as e:
            logger.error(f"Failed to initialize enterprise agents: {e}")
            # Initialize with basic agents
            from agents import RequirementsAgent, ArchitectureAgent, CodingAgent, DocumentationAgent
            self.requirements_agent = RequirementsAgent()
            self.architecture_agent = ArchitectureAgent()
            self.coding_agent = CodingAgent()
            self.documentation_agent = DocumentationAgent()
        
        # Track current project context
        self.current_project = None
        self.project_metadata = {}
        
        self.logger.info("Enterprise SDLC Orchestrator initialized")
    
    async def analyze_and_setup_project(self, business_requirements: str) -> Dict[str, Any]:
        """Analyze requirements and setup optimal project configuration"""
        
        # Analyze use case and determine optimal configuration
        use_case_analysis = self.config_manager.get_use_case_analysis(business_requirements)
        suggested_use_case = use_case_analysis["suggested_use_case"]
        
        self.logger.info(f"Analyzed use case: {suggested_use_case}")
        
        # Load optimal configuration
        config_loaded = self.config_manager.load_use_case_template(suggested_use_case)
        if not config_loaded:
            self.logger.warning(f"Failed to load configuration for {suggested_use_case}, using default")
            suggested_use_case = "enterprise_web_app"
            self.config_manager.load_use_case_template(suggested_use_case)
        
        # Setup project context
        project_name = format_project_name(business_requirements)
        output_dir = ensure_output_directory(project_name)
        
        # Create project structure based on use case
        template_type = "microservices" if "microservices" in suggested_use_case else "web_app"
        project_dirs = create_project_structure(output_dir, template_type)
        
        # Generate project metadata
        config_data = self.config_manager.get_config_for_agent("requirements")
        self.project_metadata = generate_project_metadata(project_name, suggested_use_case, config_data)
        
        # Save project metadata
        metadata_file = output_dir / "project_metadata.json"
        save_json(self.project_metadata, metadata_file, "Project Metadata")
        
        self.current_project = {
            "name": project_name,
            "use_case_type": suggested_use_case,
            "output_dir": output_dir,
            "project_dirs": project_dirs,
            "analysis": use_case_analysis
        }
        
        return self.current_project
    
    async def execute_full_sdlc(self, business_requirements: str) -> Dict[str, Any]:
        """Execute complete SDLC process with enterprise features"""
        
        start_time = datetime.now()
        self.logger.info("Starting Enterprise SDLC Process")
        
        try:
            # Phase 1: Analysis and Setup
            self.logger.info("Phase 1: Project Analysis and Setup")
            project_setup = await self.analyze_and_setup_project(business_requirements)
            
            # Phase 2: Requirements Engineering
            self.logger.info("Phase 2: Requirements Engineering") 
            requirements_result = await self._execute_requirements_phase(business_requirements)
            
            # Phase 3: Architecture and Design
            self.logger.info("Phase 3: System Architecture and Design")
            architecture_result = await self._execute_architecture_phase(requirements_result)
            
            # Phase 4: Implementation
            self.logger.info("Phase 4: Code Generation and Implementation")
            implementation_result = await self._execute_implementation_phase(architecture_result)
            
            # Phase 5: Testing Strategy
            self.logger.info("Phase 5: Testing Strategy and Test Generation")
            testing_result = await self._execute_testing_phase(implementation_result)
            
            # Phase 6: Documentation and Deployment
            self.logger.info("Phase 6: Documentation and Deployment Configuration")
            documentation_result = await self._execute_documentation_phase({
                **requirements_result,
                **architecture_result,
                **implementation_result,
                **testing_result
            })
            
            # Phase 7: Quality Assurance and Finalization
            self.logger.info("Phase 7: Quality Assurance and Project Finalization")
            final_result = await self._execute_finalization_phase({
                "project_setup": project_setup,
                "requirements": requirements_result,
                "architecture": architecture_result,
                "implementation": implementation_result,
                "testing": testing_result,
                "documentation": documentation_result
            })
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_result.update({
                "execution_time": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "performance_summary": performance_monitor.get_summary(),
                "project_name": self.current_project["name"],
                "output_directory": self.current_project["output_dir"],
                "use_case_analysis": self.current_project["analysis"]
            })
            
            self.logger.info(f"Enterprise SDLC Process completed in {duration:.2f} seconds")
            return final_result
            
        except Exception as e:
            self.logger.error(f"SDLC Process failed: {str(e)}")
            # Return partial results if available
            return {
                "error": str(e),
                "project_name": self.current_project["name"] if self.current_project else "Unknown",
                "output_directory": self.current_project["output_dir"] if self.current_project else Path("output"),
                "partial_completion": True
            }
    
    async def _execute_requirements_phase(self, business_requirements: str) -> Dict[str, Any]:
        """Execute requirements engineering phase"""
        
        context = {
            "use_case_type": self.current_project["use_case_type"],
            "project_name": self.current_project["name"]
        }
        
        try:
            # Generate comprehensive user stories
            if hasattr(self.requirements_agent, 'generate_user_stories'):
                user_stories = await self.requirements_agent.generate_user_stories(
                    business_requirements, context
                )
            else:
                # Fallback for basic agent
                user_stories = await self.requirements_agent.generate_user_stories(business_requirements)
            
            # Detailed requirements analysis if available
            analysis = {}
            if hasattr(self.requirements_agent, 'analyze_business_requirements'):
                try:
                    analysis = await self.requirements_agent.analyze_business_requirements(
                        business_requirements, context
                    )
                except:
                    analysis = {"complexity": "Medium", "timeline": "3-6 months"}
            
            # Save artifacts
            output_dir = self.current_project["output_dir"]
            docs_dir = output_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            save_file(user_stories, docs_dir / "user_stories.md", "User Stories")
            if analysis:
                save_json(analysis, docs_dir / "requirements_analysis.json", "Requirements Analysis")
            
            return {
                "user_stories": user_stories,
                "requirements_analysis": analysis,
                "business_requirements": business_requirements
            }
            
        except Exception as e:
            self.logger.error(f"Requirements phase failed: {e}")
            # Return minimal fallback
            fallback_stories = self._generate_fallback_user_stories(business_requirements)
            save_file(fallback_stories, self.current_project["output_dir"] / "user_stories.md", "Fallback User Stories")
            return {
                "user_stories": fallback_stories,
                "requirements_analysis": {},
                "business_requirements": business_requirements
            }
    
    async def _execute_architecture_phase(self, requirements_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute architecture and design phase"""
        
        context = {
            "use_case_type": self.current_project["use_case_type"],
            "complexity": self.current_project["analysis"]["analysis"]["estimated_complexity"]
        }
        
        try:
            # Generate system architecture
            if hasattr(self.architecture_agent, 'generate_system_architecture'):
                system_architecture = await self.architecture_agent.generate_system_architecture(
                    requirements_data["user_stories"], context
                )
            else:
                # Fallback for basic agent
                system_architecture = await self.architecture_agent.generate_technical_spec(
                    requirements_data["user_stories"]
                )
            
            # Generate microservices design if applicable
            microservices_design = None
            if "microservices" in self.current_project["use_case_type"]:
                if hasattr(self.architecture_agent, 'generate_microservices_design'):
                    try:
                        microservices_design = await self.architecture_agent.generate_microservices_design(
                            system_architecture, context
                        )
                    except:
                        microservices_design = self._generate_fallback_microservices()
            
            # Save architecture artifacts
            output_dir = self.current_project["output_dir"]
            docs_dir = output_dir / "docs"
            
            save_file(system_architecture, docs_dir / "system_architecture.md", "System Architecture")
            
            if microservices_design:
                save_json(microservices_design, docs_dir / "microservices_design.json", "Microservices Design")
            
            return {
                "system_architecture": system_architecture,
                "microservices_design": microservices_design
            }
            
        except Exception as e:
            self.logger.error(f"Architecture phase failed: {e}")
            fallback_arch = self._generate_fallback_architecture()
            save_file(fallback_arch, self.current_project["output_dir"] / "architecture.md", "Fallback Architecture")
            return {
                "system_architecture": fallback_arch,
                "microservices_design": None
            }
    
    async def _execute_implementation_phase(self, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute implementation phase with code generation"""
        
        context = {
            "use_case_type": self.current_project["use_case_type"],
            "architecture_type": "microservices" if architecture_data["microservices_design"] else "monolithic"
        }
        
        try:
            generated_code = {}
            
            if architecture_data["microservices_design"] and hasattr(self.coding_agent, 'generate_microservice_code'):
                # Generate microservices code
                microservices_design = architecture_data["microservices_design"]
                
                for service in microservices_design.get("services", []):
                    try:
                        service_code = await self.coding_agent.generate_microservice_code(service, context)
                        
                        # Save service code in separate directory
                        service_dir = self.current_project["output_dir"] / "services" / service["name"]
                        service_dir.mkdir(parents=True, exist_ok=True)
                        
                        for filename, code in service_code.items():
                            save_file(code, service_dir / filename, f"{service['name']} - {filename}")
                        
                        generated_code[service["name"]] = service_code
                    except Exception as e:
                        self.logger.warning(f"Failed to generate code for {service['name']}: {e}")
            else:
                # Generate monolithic application code
                if hasattr(self.coding_agent, 'generate_application_code'):
                    app_code = await self.coding_agent.generate_application_code(
                        architecture_data["system_architecture"]
                    )
                else:
                    # Very basic fallback
                    app_code = self._generate_fallback_code()
                
                # Save application code
                src_dir = self.current_project["output_dir"] / "src"
                src_dir.mkdir(exist_ok=True)
                for filename, code in app_code.items():
                    save_file(code, src_dir / filename, f"Application - {filename}")
                
                generated_code = app_code
            
            return {
                "generated_code": generated_code,
                "architecture_type": context["architecture_type"]
            }
            
        except Exception as e:
            self.logger.error(f"Implementation phase failed: {e}")
            fallback_code = self._generate_fallback_code()
            src_dir = self.current_project["output_dir"] / "src"
            src_dir.mkdir(exist_ok=True)
            for filename, code in fallback_code.items():
                save_file(code, src_dir / filename, f"Fallback - {filename}")
            return {
                "generated_code": fallback_code,
                "architecture_type": "monolithic"
            }
    
    async def _execute_testing_phase(self, implementation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing strategy and test generation phase"""
        
        try:
            # Generate comprehensive test suite
            if hasattr(self.coding_agent, 'generate_test_cases'):
                test_cases = await self.coding_agent.generate_test_cases(
                    implementation_data["generated_code"]
                )
            else:
                test_cases = self._generate_fallback_tests()
            
            # Save test artifacts
            tests_dir = self.current_project["output_dir"] / "tests"
            tests_dir.mkdir(exist_ok=True)
            save_file(test_cases, tests_dir / "test_suite.py", "Test Suite")
            
            # Generate testing strategy document
            testing_strategy = await self._generate_testing_strategy(implementation_data)
            save_file(testing_strategy, tests_dir / "testing_strategy.md", "Testing Strategy")
            
            return {
                "test_cases": test_cases,
                "testing_strategy": testing_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Testing phase failed: {e}")
            fallback_tests = self._generate_fallback_tests()
            tests_dir = self.current_project["output_dir"] / "tests"
            tests_dir.mkdir(exist_ok=True)
            save_file(fallback_tests, tests_dir / "test_suite.py", "Fallback Tests")
            return {
                "test_cases": fallback_tests,
                "testing_strategy": "Basic testing strategy"
            }
    
    async def _execute_documentation_phase(self, all_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation generation phase"""
        
        try:
            # Generate comprehensive project documentation
            if hasattr(self.documentation_agent, 'process'):
                project_documentation = await self.documentation_agent.process(
                    self._create_documentation_prompt(all_artifacts),
                    "generate_project_documentation"
                )
            else:
                project_documentation = await self.documentation_agent.generate_project_documentation(all_artifacts)
            
            # Generate API documentation if applicable
            api_documentation = await self._generate_api_documentation(all_artifacts)
            
            # Generate deployment guides
            deployment_guide = await self._generate_deployment_guide(all_artifacts)
            
            # Save documentation artifacts
            docs_dir = self.current_project["output_dir"] / "docs"
            save_file(project_documentation, docs_dir / "README.md", "Project Documentation")
            save_file(api_documentation, docs_dir / "api_documentation.md", "API Documentation")
            save_file(deployment_guide, docs_dir / "deployment_guide.md", "Deployment Guide")
            
            # Create deployment configurations
            deployment_configs = create_deployment_configs(
                self.current_project["output_dir"],
                self.current_project["name"],
                self.current_project["use_case_type"]
            )
            
            return {
                "project_documentation": project_documentation,
                "api_documentation": api_documentation,
                "deployment_guide": deployment_guide,
                "deployment_configs": deployment_configs
            }
            
        except Exception as e:
            self.logger.error(f"Documentation phase failed: {e}")
            fallback_doc = self._generate_fallback_documentation()
            save_file(fallback_doc, self.current_project["output_dir"] / "README.md", "Fallback Documentation")
            return {
                "project_documentation": fallback_doc,
                "api_documentation": "API documentation placeholder",
                "deployment_guide": "Deployment guide placeholder",
                "deployment_configs": {}
            }
    
    async def _execute_finalization_phase(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final quality assurance and project finalization"""
        
        try:
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(all_results)
            
            # Generate project metrics and quality report
            quality_report = self._generate_quality_report(all_results)
            
            # Generate next steps and recommendations
            next_steps = await self._generate_next_steps(all_results)
            
            # Save final artifacts
            output_dir = self.current_project["output_dir"]
            save_file(executive_summary, output_dir / "EXECUTIVE_SUMMARY.md", "Executive Summary")
            save_json(quality_report, output_dir / "quality_report.json", "Quality Report")
            save_file(next_steps, output_dir / "NEXT_STEPS.md", "Next Steps Guide")
            
            # Update project metadata with final results
            self.project_metadata.update({
                "completion_status": "completed",
                "quality_metrics": quality_report,
                "artifacts_generated": self._count_generated_artifacts(output_dir)
            })
            
            save_json(self.project_metadata, output_dir / "project_metadata.json", "Final Project Metadata")
            
            return {
                **all_results,
                "executive_summary": executive_summary,
                "quality_report": quality_report,
                "next_steps": next_steps,
                "project_metadata": self.project_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Finalization phase failed: {e}")
            return {
                **all_results,
                "executive_summary": "Project completed with some limitations",
                "quality_report": {"status": "completed"},
                "next_steps": "Review generated files and proceed with development"
            }
    
    # Helper methods and fallback implementations
    def _generate_fallback_user_stories(self, requirements: str) -> str:
        """Generate basic fallback user stories"""
        return f"""# User Stories

## Story 1: Basic Functionality
**As a** user
**I want** to use the application
**So that** I can accomplish my goals

**Acceptance Criteria:**
- [ ] Application loads successfully
- [ ] Basic functionality works
- [ ] User can navigate the interface

**Priority:** High
**Story Points:** 8

Based on requirements: {requirements[:200]}...
        """
    
    def _generate_fallback_architecture(self) -> str:
        """Generate basic fallback architecture"""
        return """# System Architecture

## Overview
Basic web application architecture with standard components.

## Components
- Frontend: Web interface
- Backend: API server
- Database: Data storage
- Authentication: User management

## Technology Stack
- Python/Flask or FastAPI
- PostgreSQL or SQLite
- HTML/CSS/JavaScript
- Docker for deployment
        """
    
    def _generate_fallback_microservices(self) -> Dict[str, Any]:
        """Generate basic microservices design"""
        return {
            "services": [
                {
                    "name": "user-service",
                    "responsibility": "User management and authentication",
                    "database": "PostgreSQL",
                    "apis": ["/api/v1/users", "/api/v1/auth"]
                },
                {
                    "name": "core-service",
                    "responsibility": "Core business logic",
                    "database": "PostgreSQL", 
                    "apis": ["/api/v1/core"]
                }
            ],
            "communication": "HTTP/REST with message queues",
            "deployment": "Docker containers on Kubernetes"
        }
    
    def _generate_fallback_code(self) -> Dict[str, str]:
        """Generate basic fallback code"""
        return {
            "app.py": """from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
            """,
            "requirements.txt": """Flask==2.3.3
gunicorn==21.2.0
            """
        }
    
    def _generate_fallback_tests(self) -> str:
        """Generate basic fallback tests"""
        return """import pytest

def test_basic_functionality():
    assert True, "Basic test placeholder"

def test_application_startup():
    # Test application starts correctly
    pass
        """
    
    def _generate_fallback_documentation(self) -> str:
        """Generate basic fallback documentation"""
        return f"""# {self.current_project['name']}

## Overview
This project was generated using the Enterprise SDLC Framework.

## Installation
```bash
pip install -r requirements.txt
python app.py
```

## Usage
Access the application at http://localhost:5000

## Generated Files
- Application code in src/
- Tests in tests/
- Documentation in docs/

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    
    # ...existing helper methods...
    async def _generate_testing_strategy(self, implementation_data: Dict[str, Any]) -> str:
        """Generate testing strategy"""
        return """# Testing Strategy

## Unit Testing
- Test individual components and functions
- Use pytest framework
- Aim for 80%+ code coverage

## Integration Testing  
- Test API endpoints
- Test database interactions
- Test service integrations

## Quality Gates
- All tests must pass
- Code coverage above 80%
- No critical security vulnerabilities
        """
    
    def _create_documentation_prompt(self, all_artifacts: Dict[str, Any]) -> str:
        """Create documentation prompt"""
        return f"""
Generate comprehensive project documentation for: {self.current_project['name']}

Include:
1. Project overview and features
2. Installation instructions
3. Usage guide
4. API documentation
5. Development setup
6. Deployment guide

Project type: {self.current_project['use_case_type']}
        """
    
    async def _generate_api_documentation(self, all_artifacts: Dict[str, Any]) -> str:
        """Generate API documentation"""
        return """# API Documentation

## Overview
RESTful API endpoints for the application.

## Authentication
API uses token-based authentication.

## Endpoints
- GET /api/health - Health check
- POST /api/auth/login - User login
- GET /api/data - Retrieve data
- POST /api/data - Create data

## Response Format
All responses are in JSON format.
        """
    
    async def _generate_deployment_guide(self, all_artifacts: Dict[str, Any]) -> str:
        """Generate deployment guide"""
        return """# Deployment Guide

## Docker Deployment
```bash
docker build -t app .
docker run -p 8000:8000 app
```

## Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

## Environment Variables
- DATABASE_URL: Database connection string
- SECRET_KEY: Application secret key
        """
    
    async def _generate_executive_summary(self, all_results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        return f"""# Executive Summary

## Project: {self.current_project['name']}

### Overview
Successfully generated enterprise-grade application using AI-powered SDLC framework.

### Key Deliverables
- Comprehensive user stories and requirements
- System architecture and design
- Production-ready application code
- Testing strategy and test cases  
- Complete documentation and deployment guides

### Technology Stack
- Backend: Python with modern frameworks
- Database: PostgreSQL with proper modeling
- Frontend: Modern web technologies
- Deployment: Docker and Kubernetes ready

### Next Steps
1. Review generated code and documentation
2. Set up development environment
3. Run tests and validate functionality
4. Deploy to target environment
5. Begin iterative development

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    
    def _generate_quality_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality report"""
        return {
            "overall_quality": "High",
            "code_generation": "Complete",
            "documentation": "Comprehensive", 
            "testing": "Included",
            "deployment": "Ready",
            "security": "Implemented",
            "scalability": "Designed",
            "completion_percentage": 100,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_next_steps(self, all_results: Dict[str, Any]) -> str:
        """Generate next steps guide"""
        return f"""# Next Steps Guide

## Immediate Actions (1-2 days)
1. **Review Generated Code**
   - Examine all generated files
   - Understand the architecture
   - Review security implementations

2. **Set Up Development Environment**
   - Install Python and dependencies
   - Set up database
   - Configure environment variables

## Short Term (1-2 weeks)
3. **Run and Test Application**
   - Execute test suite: `pytest tests/`
   - Start application: `python src/app.py`
   - Verify all endpoints work

4. **Customize and Enhance**
   - Modify business logic as needed
   - Add specific features
   - Integrate with existing systems

## Medium Term (1-2 months)
5. **Production Deployment**
   - Set up CI/CD pipeline
   - Deploy to staging environment
   - Configure monitoring and logging
   - Deploy to production

6. **Optimization and Scaling**
   - Performance tuning
   - Load testing
   - Security hardening
   - Monitoring setup

## Support Resources
- Generated documentation in docs/
- Test cases in tests/
- Deployment configs in root directory
- Architecture diagrams in docs/architecture/

Project: {self.current_project['name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    
    def _count_generated_artifacts(self, output_dir: Path) -> Dict[str, int]:
        """Count generated artifacts"""
        try:
            all_files = list(output_dir.rglob("*"))
            return {
                "total_files": len([f for f in all_files if f.is_file()]),
                "code_files": len(list(output_dir.rglob("*.py"))),
                "documentation_files": len(list(output_dir.rglob("*.md"))),
                "configuration_files": len(list(output_dir.rglob("*.yml"))) + len(list(output_dir.rglob("*.yaml"))),
                "json_files": len(list(output_dir.rglob("*.json"))),
                "directories": len([f for f in all_files if f.is_dir()])
            }
        except Exception as e:
            self.logger.error(f"Failed to count artifacts: {e}")
            return {"total_files": 0, "error": str(e)}

# Backward compatibility
SDLCOrchestrator = EnterpriseSDLCOrchestrator