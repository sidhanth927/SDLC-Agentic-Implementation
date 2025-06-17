import asyncio
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager
from agents import EnterpriseRequirementsAgent, EnterpriseArchitectureAgent, EnterpriseCodingAgent, EnterpriseBaseAgent
from utils import setup_logging, ensure_output_directory, save_file, format_project_name
from performance_monitor import performance_monitor
from config import config

# Try to import the enterprise orchestrator, fallback to simple orchestrator
try:
    from orchestrator import EnterpriseSDLCOrchestrator as SDLCOrchestrator
except ImportError:
    logger = setup_logging()
    logger.warning("Enterprise orchestrator not found, using simplified orchestrator")
    
    class SDLCOrchestrator:
        """Simplified orchestrator for compatibility"""
        
        def __init__(self):
            self.logger = setup_logging()
            
            # Initialize agents with enterprise versions
            self.requirements_agent = EnterpriseRequirementsAgent()
            self.architecture_agent = EnterpriseArchitectureAgent()
            self.coding_agent = EnterpriseCodingAgent()
            self.documentation_agent = EnterpriseBaseAgent("documentation")
            
            self.logger.info("Simplified SDLC Orchestrator initialized")
        
        async def execute_full_sdlc(self, business_requirements: str) -> dict:
            """Execute simplified SDLC process"""
            project_name = format_project_name(business_requirements)
            output_dir = ensure_output_directory(project_name)
            
            self.logger.info(f"Starting SDLC process for project: {project_name}")
            
            results = {
                "project_name": project_name,
                "output_directory": output_dir,
                "business_requirements": business_requirements
            }
            
            # Analyze use case and configure
            use_case_analysis = config.get_use_case_analysis(business_requirements)
            suggested_use_case = use_case_analysis["suggested_use_case"]
            config.load_use_case_template(suggested_use_case)
            
            self.logger.info(f"Detected use case: {suggested_use_case}")
            
            # Execute phases
            context = {"use_case_type": suggested_use_case, "project_name": project_name}
            
            # Phase 1: Requirements
            user_stories = await self.requirements_agent.generate_user_stories(business_requirements, context)
            save_file(user_stories, output_dir / f"{project_name}_user_stories.md", "User Stories")
            
            # Phase 2: Architecture  
            system_architecture = await self.architecture_agent.generate_system_architecture(user_stories, context)
            save_file(system_architecture, output_dir / f"{project_name}_architecture.md", "System Architecture")
            
            # Phase 3: Code Generation
            if "microservices" in suggested_use_case:
                # Generate microservices
                microservices_design = await self.architecture_agent.generate_microservices_design(system_architecture, context)
                
                for service in microservices_design.get("services", []):
                    service_code = await self.coding_agent.generate_microservice_code(service, context)
                    service_dir = output_dir / "services" / service["name"]
                    service_dir.mkdir(parents=True, exist_ok=True)
                    
                    for filename, code in service_code.items():
                        save_file(code, service_dir / filename, f"{service['name']} - {filename}")
            else:
                # Generate monolithic application
                application_code = await self.coding_agent.generate_application_code(system_architecture)
                for filename, code in application_code.items():
                    save_file(code, output_dir / filename, f"Application - {filename}")
            
            # Phase 4: Documentation
            documentation = await self.documentation_agent.process(
                f"Generate comprehensive documentation for {project_name} based on the generated artifacts.",
                "generate_documentation"
            )
            save_file(documentation, output_dir / f"{project_name}_README.md", "Project Documentation")
            
            results.update({
                "user_stories": user_stories,
                "system_architecture": system_architecture,
                "documentation": documentation,
                "use_case_analysis": use_case_analysis
            })
            
            return results

logger = setup_logging()

async def main():
    """Enhanced main entry point with enterprise features"""
    try:
        print("🚀 Enterprise Multi-Agent SDLC Automation Framework v2.0")
        print("=" * 65)
        
        # Display system information
        model_info = ModelManager.get_comprehensive_info()
        print(f"System Information:")
        print(f"- CUDA Available: {model_info['cuda_available']}")
        print(f"- GPU Memory: {model_info.get('gpu_memory', 0):.1f} GB")
        print(f"- Available RAM: {model_info.get('available_ram', 0):.1f} GB")
        print(f"- Runnable Models: {model_info['runnable_models']}/{model_info['total_models']}")
        print()
        
        # Initialize orchestrator
        orchestrator = SDLCOrchestrator()
        
        # Get business requirements
        business_requirements = await get_business_requirements()
        
        if not business_requirements:
            print("❌ No business requirements provided. Exiting.")
            return
        
        print(f"\n📊 Processing requirements ({len(business_requirements)} characters)...")
        print("This may take several minutes depending on system capabilities and use case complexity.")
        print()
        
        # Execute SDLC process
        start_time = datetime.now()
        
        if hasattr(orchestrator, 'execute_full_sdlc'):
            results = await orchestrator.execute_full_sdlc(business_requirements)
        else:
            # Fallback to simplified process
            results = await orchestrator.process_business_requirements(business_requirements)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        display_results(results, duration)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Detailed error information:")
        print("\n💡 Troubleshooting tips:")
        print("- Ensure all dependencies are installed: pip install -r requirements.txt")
        print("- Check your internet connection for API models")
        print("- Try using local models if API keys are not configured")
        print("- Verify sufficient system resources (RAM/GPU)")

async def get_business_requirements() -> str:
    """Get business requirements from user input"""
    print("📋 Please provide your business requirements:")
    print("Choose an option:")
    print("1. Enter requirements manually")
    print("2. Use Text-to-SQL example")
    print("3. Use E-commerce example") 
    print("4. Use Microservices platform example")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\nEnter your requirements (press Ctrl+D on Linux/Mac or Ctrl+Z on Windows when done):")
            requirements = []
            try:
                while True:
                    line = input()
                    requirements.append(line)
            except EOFError:
                pass
            return '\n'.join(requirements).strip()
            
        elif choice == "2":
            return get_text_to_sql_example()
        elif choice == "3":
            return get_ecommerce_example()
        elif choice == "4":
            return get_microservices_example()
        else:
            print("Invalid choice, using Text-to-SQL example...")
            return get_text_to_sql_example()
            
    except KeyboardInterrupt:
        print("\nUsing default Text-to-SQL example...")
        return get_text_to_sql_example()

def get_text_to_sql_example() -> str:
    """Get Text-to-SQL example requirements"""
    return """
# Text-To-SQL Intelligent Query Generator

We need an advanced Text-To-SQL application that leverages Large Language Models to convert natural language queries into accurate SQL statements for database operations.

## Core Features:

### Natural Language Processing:
- Accept natural language queries in multiple formats and complexity levels
- Support conversational queries with context awareness
- Handle ambiguous queries with clarification prompts
- Process complex business logic questions into SQL
- Support multiple languages (English, Spanish, French, German)
- Query intent recognition and classification
- Auto-correction for common linguistic errors

### SQL Generation & Optimization:
- Generate syntactically correct SQL for multiple database engines (MySQL, PostgreSQL, SQLite, SQL Server, Oracle)
- Support complex queries including JOINs, subqueries, window functions, and CTEs
- Query optimization suggestions and performance hints
- Generate both SELECT and modification queries (INSERT, UPDATE, DELETE)
- Handle aggregate functions, grouping, and statistical operations

### Database Integration:
- Connect to multiple database types simultaneously
- Database schema introspection and metadata extraction
- Table relationship inference and foreign key analysis
- Real-time database connection validation
- Query execution with result preview and pagination

### User Interface Features:
- Web-based interface with modern, intuitive design
- Query history and favorites management
- Interactive query builder with drag-and-drop capabilities
- Syntax highlighting for generated SQL
- Real-time query validation and error detection
- Export results in multiple formats (CSV, JSON, Excel, PDF)

### Advanced AI Features:
- Context-aware query suggestions based on database schema
- Learning from user corrections and feedback
- Query explanation in natural language
- Alternative query suggestions for optimization
- Automated data insights and pattern recognition

## Technical Requirements:

### LLM Integration:
- Support for multiple LLM providers (OpenAI GPT, Anthropic Claude, Google Gemini, local models)
- Fine-tuned models for SQL generation tasks
- Model switching based on query complexity
- Token usage optimization and cost management

### Architecture:
- Microservices architecture with containerized deployment
- RESTful API for all functionalities
- WebSocket support for real-time query execution
- Asynchronous query processing for large datasets
- Caching layer for frequent queries and schema metadata

### Security & Compliance:
- Data encryption at rest and in transit
- SQL injection prevention and query sanitization
- PII detection and masking capabilities
- GDPR and compliance features
- Rate limiting and DOS protection

### Performance & Scalability:
- Sub-second response time for simple queries
- Concurrent user support (1000+ simultaneous users)
- Query result caching and intelligent cache invalidation
- Auto-scaling based on load patterns

The application should handle enterprise-scale databases with millions of records, support complex analytical queries, and provide an intuitive interface that makes database querying accessible to both technical and non-technical users.
    """.strip()

def get_ecommerce_example() -> str:
    """Get E-commerce example requirements"""
    return """
# E-Commerce Platform

We need a comprehensive e-commerce platform that provides a complete online shopping experience.

## Core Features:

### Product Management:
- Product catalog with categories, subcategories, and tags
- Product variants (size, color, style) with inventory tracking
- High-resolution image galleries with zoom functionality
- Product reviews and ratings system
- Related and recommended products
- Product comparison features
- Bulk product import/export for merchants

### Shopping Experience:
- Advanced search with filters (price, brand, category, ratings)
- Shopping cart with save for later functionality
- Wishlist and favorites management
- Guest checkout and registered user checkout
- Multiple payment methods (credit cards, PayPal, digital wallets)
- Real-time inventory updates
- Mobile-responsive design

### User Management:
- Customer registration and profile management
- Order history and tracking
- Address book management
- Merchant/seller onboarding and dashboard
- Role-based access control (customers, merchants, admins)
- Social login integration

### Order Management:
- Order processing workflow
- Inventory management and stock alerts
- Shipping integration with multiple carriers
- Order tracking and notifications
- Return and refund management
- Invoice generation

## Technical Requirements:

### Performance:
- Handle 10,000+ concurrent users
- Sub-3-second page load times
- Real-time inventory synchronization
- CDN integration for global content delivery

### Security:
- PCI DSS compliance for payment processing
- Data encryption and secure transactions
- Fraud detection and prevention
- GDPR compliance for data protection

### Integration:
- Payment gateway integration (Stripe, PayPal, etc.)
- Shipping carrier APIs
- Email marketing platform integration
- Analytics and reporting tools
- Social media integration

### Scalability:
- Microservices architecture
- Database optimization for high-volume transactions
- Caching strategies for product catalogs
- Auto-scaling infrastructure

The platform should support both B2C and B2B scenarios, with multi-vendor capabilities and comprehensive analytics for business insights.
    """.strip()

def get_microservices_example() -> str:
    """Get Microservices platform example requirements"""
    return """
# Enterprise Microservices Platform

We need a comprehensive microservices platform that enables organizations to build, deploy, and manage distributed applications at scale.

## Core Platform Features:

### Service Management:
- Service discovery and registration
- Service mesh integration with Istio/Envoy
- API gateway with routing and load balancing
- Service versioning and blue-green deployments
- Circuit breaker and bulkhead patterns
- Distributed tracing and observability

### Development Tools:
- Service template generation
- Local development environment setup
- Testing frameworks for microservices
- API documentation generation
- Contract testing between services
- Development workflow automation

### Infrastructure Management:
- Kubernetes orchestration
- Container registry management
- Infrastructure as Code (Terraform/Helm)
- Multi-cloud deployment support
- Resource quota and limit management
- Secrets and configuration management

### Monitoring & Observability:
- Centralized logging aggregation
- Metrics collection and visualization
- Distributed tracing with Jaeger/Zipkin
- Health checks and alerting
- Performance monitoring and profiling
- Security scanning and compliance monitoring

### Developer Experience:
- Self-service platform for developers
- CI/CD pipeline automation
- Code quality gates and security scanning
- Documentation portal
- Developer onboarding workflows
- Team collaboration tools

## Technical Requirements:

### Architecture:
- Event-driven architecture with message queues
- CQRS and Event Sourcing patterns
- Database per service pattern
- Saga pattern for distributed transactions
- API-first design principles

### Security:
- Zero-trust security model
- Service-to-service authentication (mTLS)
- OAuth2/JWT token management
- Network policies and segmentation
- Vulnerability scanning and compliance

### Scalability:
- Horizontal auto-scaling
- Multi-region deployment
- Cache-aside pattern implementation
- Database sharding strategies
- Event streaming with Kafka

### Reliability:
- 99.99% availability target
- Disaster recovery and backup strategies
- Chaos engineering practices
- Progressive delivery mechanisms
- Automated rollback capabilities

The platform should enable teams to independently develop, test, and deploy services while maintaining system-wide consistency, security, and reliability standards.
    """.strip()

def display_results(results: dict, duration: float):
    """Display execution results"""
    print("\n" + "=" * 65)
    print("✅ Enterprise SDLC Process Completed Successfully!")
    print(f"⏱️  Total execution time: {duration:.1f} seconds")
    print(f"📁 Output directory: {results['output_directory']}")
    
    # Display use case analysis if available
    if 'use_case_analysis' in results:
        analysis = results['use_case_analysis']
        print(f"🎯 Detected use case: {analysis['suggested_use_case']}")
        print(f"📊 Complexity: {analysis['analysis']['estimated_complexity']}")
        print(f"⏰ Estimated timeline: {analysis['analysis']['estimated_timeline']}")
        print(f"👥 Recommended team size: {analysis['analysis']['team_size_recommendation']}")
    
    # Display generated files
    output_dir = Path(results['output_directory'])
    files = list(output_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    print(f"\n📄 Generated {len([f for f in files if f.is_file()])} files:")
    for file_path in sorted(output_dir.glob("*")):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            print(f"   ✓ {file_path.name} ({file_size:,} bytes)")
        elif file_path.is_dir() and any(file_path.iterdir()):
            sub_files = len(list(file_path.rglob("*")))
            print(f"   📁 {file_path.name}/ ({sub_files} files)")
    
    print(f"\n📊 Total output: {total_size:,} bytes")
    
    # Display next steps
    print("\n🚀 Next Steps:")
    print("1. Review generated files and architecture")
    print("2. Set up development environment")
    print("3. Install dependencies and run tests")
    print("4. Deploy services to your infrastructure")
    print("5. Configure monitoring and observability")
    
    # Display performance summary
    perf_summary = performance_monitor.get_summary()
    if perf_summary.strip():
        print("\n📈 Performance Summary:")
        for line in perf_summary.split('\n'):
            if line.strip():
                print(f"   {line}")

def run_interactive_mode():
    """Enhanced interactive mode with enterprise features"""
    print("🤖 Enterprise SDLC Framework - Interactive Mode")
    print("Enhanced with microservices, enterprise patterns, and multiple LLM support")
    print("Type 'help' for commands or 'exit' to quit")
    
    orchestrator = None
    
    while True:
        try:
            command = input("\nEnterprise-SDLC> ").strip().lower()
            
            if command == 'exit':
                print("👋 Goodbye!")
                break
            elif command == 'help':
                print("""
Available commands:
- init: Initialize the enterprise framework
- status: Show system status and model information
- process: Process business requirements with enterprise features
- templates: Show available use case templates
- models: Show available models and capabilities
- config: Show current configuration
- metrics: Show performance metrics
- demo: Run demonstration with predefined examples
- help: Show this help message
- exit: Exit the framework
                """)
            elif command == 'init':
                print("🔧 Initializing enterprise framework...")
                orchestrator = SDLCOrchestrator()
                print("✅ Enterprise framework initialized successfully")
            elif command == 'templates':
                print("📋 Available Use Case Templates:")
                for name, template in config.USE_CASE_TEMPLATES.items():
                    print(f"- {name}: {template.description} ({template.complexity})")
            elif command == 'models':
                model_info = ModelManager.get_comprehensive_info()
                print("🤖 Available Models by Capability:")
                for capability, models in model_info['available_models'].items():
                    print(f"- {capability}: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            elif command == 'config':
                print("⚙️ Current Configuration:")
                print(config.get_system_summary())
            elif command == 'demo':
                if not orchestrator:
                    orchestrator = SDLCOrchestrator()
                print("🎯 Running enterprise demo...")
                asyncio.run(orchestrator.execute_full_sdlc(get_text_to_sql_example()))
            # ...existing code...
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def run_enterprise_demo():
    """Run enterprise demonstration"""
    print("🎯 Enterprise SDLC Framework Demo")
    print("=" * 65)
    
    demo_requirements = get_text_to_sql_example()
    
    print("📋 Demo: Text-to-SQL Intelligent Query Generator")
    print("This demonstration showcases enterprise-grade microservices generation")
    print("\n" + "=" * 65)
    
    orchestrator = SDLCOrchestrator()
    
    start_time = datetime.now()
    results = await orchestrator.execute_full_sdlc(demo_requirements)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    display_results(results, duration)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            run_interactive_mode()
        elif sys.argv[1] == "--demo":
            asyncio.run(run_enterprise_demo())
        elif sys.argv[1] == "--help":
            print("""
Enterprise SDLC Framework v2.0

Usage: python main.py [options]

Options:
  --interactive    Run in interactive mode
  --demo          Run enterprise demonstration
  --help          Show this help message

Examples:
  python main.py                    # Run main interface
  python main.py --interactive      # Interactive command-line mode
  python main.py --demo            # Run Text-to-SQL demo
            """)
        else:
            print("Unknown option. Use --help for usage information.")
            sys.exit(1)
    else:
        asyncio.run(main())