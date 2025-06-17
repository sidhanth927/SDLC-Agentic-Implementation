import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import re
import json
import yaml

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging for the framework"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sdlc_framework_{timestamp}.log"
    
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    # Return framework logger
    framework_logger = logging.getLogger("SDLCFramework")
    framework_logger.info(f"Logging initialized - Level: {log_level}")
    
    return framework_logger

def ensure_output_directory(project_name: str) -> Path:
    """Create and return output directory for project artifacts"""
    
    # Sanitize project name for filesystem
    safe_name = re.sub(r'[^\w\-_\.]', '_', project_name)
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    
    # Create output directory structure
    base_dir = Path("output")
    project_dir = base_dir / safe_name
    
    # Create directories
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "code").mkdir(exist_ok=True)
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    (project_dir / "config").mkdir(exist_ok=True)
    
    logger.info(f"Created output directory: {project_dir}")
    return project_dir

def save_file(content: str, file_path: Path, description: str = "File") -> bool:
    """Save content to file with error handling"""
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = file_path.stat().st_size
        logger.info(f"Saved {description}: {file_path.name} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {description} to {file_path}: {str(e)}")
        return False

def save_json(data: Dict[str, Any], file_path: Path, description: str = "JSON file") -> bool:
    """Save dictionary as JSON file"""
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return save_file(content, file_path, description)
    except Exception as e:
        logger.error(f"Failed to save JSON {description}: {str(e)}")
        return False

def save_yaml(data: Dict[str, Any], file_path: Path, description: str = "YAML file") -> bool:
    """Save dictionary as YAML file"""
    try:
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        return save_file(content, file_path, description)
    except Exception as e:
        logger.error(f"Failed to save YAML {description}: {str(e)}")
        return False

def format_project_name(business_requirements: str) -> str:
    """Extract and format project name from business requirements"""
    
    # Try to extract from first line if it looks like a title
    lines = business_requirements.strip().split('\n')
    first_line = lines[0].strip()
    
    # Check if first line looks like a title (short and descriptive)
    if (len(first_line) < 100 and 
        len(first_line) > 5 and
        not first_line.lower().startswith(('we need', 'create', 'build', 'develop'))):
        # Remove markdown headers and clean up
        clean_title = re.sub(r'^#+\s*', '', first_line)
        return clean_title
    
    # Extract from common patterns
    patterns = [
        r'(?:build|create|develop|design)\s+(?:a|an)?\s*([^\.]+?)(?:\s+(?:system|application|app|platform|tool))',
        r'([^\.]+?)(?:\s+(?:system|application|app|platform|tool))',
        r'(?:project|system|application):\s*([^\.]+)',
    ]
    
    text = business_requirements[:500].lower()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if 5 < len(name) < 50:
                return name.title()
    
    # Fallback to generic name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"Generated_Project_{timestamp}"

def validate_generated_content(content: str, content_type: str = "general") -> Dict[str, Any]:
    """Validate quality of generated content"""
    
    validation_result = {
        "valid": True,
        "quality_score": 0.0,
        "issues": [],
        "metrics": {}
    }
    
    if not content or len(content.strip()) < 20:  # Reduced threshold for local models
        validation_result["valid"] = False
        validation_result["issues"].append("Content too short")
        return validation_result
    
    # Basic metrics
    word_count = len(content.split())
    line_count = len(content.split('\n'))
    char_count = len(content)
    
    validation_result["metrics"] = {
        "word_count": word_count,
        "line_count": line_count,
        "character_count": char_count
    }
    
    # Content type specific validation
    if content_type == "user_stories":
        required_elements = ["As a", "I want", "So that"]
        story_count = content.count("As a")
        
        validation_result["metrics"]["story_count"] = story_count
        
        missing_elements = [elem for elem in required_elements if elem not in content]
        if missing_elements:
            validation_result["issues"].extend([f"Missing: {elem}" for elem in missing_elements])
        
        if story_count < 2:  # Reduced threshold
            validation_result["issues"].append("Insufficient number of user stories")
    
    elif content_type == "code":
        # Check for basic code structure
        code_indicators = ["def ", "class ", "import ", "from "]
        has_code = any(indicator in content for indicator in code_indicators)
        
        if not has_code:
            validation_result["issues"].append("Does not appear to contain valid code")
        
        # Check for basic security patterns
        security_issues = []
        if "password" in content.lower() and "hash" not in content.lower():
            security_issues.append("Potential password security issue")
        
        validation_result["metrics"]["security_issues"] = security_issues
    
    elif content_type == "documentation":
        # Check for documentation structure
        required_sections = ["#", "##"]
        missing_sections = [section for section in required_sections if section not in content]
        
        if missing_sections:
            validation_result["issues"].extend([f"Missing section: {section}" for section in missing_sections])
    
    # Calculate quality score
    base_score = min(1.0, word_count / 200)  # Reduced threshold for local models
    
    # Penalty for issues
    issue_penalty = len(validation_result["issues"]) * 0.1
    validation_result["quality_score"] = max(0.0, base_score - issue_penalty)
    
    # Mark as invalid if quality is too low
    if validation_result["quality_score"] < 0.2:  # Reduced threshold
        validation_result["valid"] = False
    
    return validation_result

def extract_code_blocks(text: str) -> Dict[str, str]:
    """Extract code blocks from markdown-formatted text"""
    
    code_blocks = {}
    
    # Pattern to match code blocks with optional language specification
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for i, (language, code) in enumerate(matches):
        # Determine filename based on language or content
        if language:
            extensions = {
                'python': '.py',
                'javascript': '.js',
                'typescript': '.ts',
                'html': '.html',
                'css': '.css',
                'sql': '.sql',
                'yaml': '.yml',
                'json': '.json'
            }
            extension = extensions.get(language.lower(), '.txt')
            filename = f"code_block_{i+1}{extension}"
        else:
            # Try to infer from content
            if 'def ' in code or 'import ' in code or 'class ' in code:
                filename = f"code_block_{i+1}.py"
            elif '<html' in code or '<!DOCTYPE' in code:
                filename = f"code_block_{i+1}.html"
            else:
                filename = f"code_block_{i+1}.txt"
        
        code_blocks[filename] = code.strip()
    
    return code_blocks

def create_project_structure(project_dir: Path, template_type: str = "web_app") -> Dict[str, Path]:
    """Create standard project directory structure"""
    
    structures = {
        "web_app": [
            "src", "src/api", "src/models", "src/services", "src/utils",
            "tests", "tests/unit", "tests/integration", "tests/api",
            "docs", "docs/api", "docs/architecture",
            "config", "scripts", "docker", "k8s"
        ],
        "microservices": [
            "services", "services/user-service", "services/auth-service", "services/core-service",
            "shared", "shared/models", "shared/utils", "shared/config",
            "infrastructure", "infrastructure/docker", "infrastructure/k8s",
            "docs", "docs/services", "docs/api", "docs/deployment",
            "tests", "tests/integration", "tests/e2e"
        ],
        "simple": [
            "src", "tests", "docs", "config"
        ]
    }
    
    structure = structures.get(template_type, structures["web_app"])
    created_dirs = {}
    
    for dir_path in structure:
        full_path = project_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        created_dirs[dir_path] = full_path
    
    logger.info(f"Created {len(created_dirs)} directories for {template_type} project structure")
    return created_dirs

def generate_project_metadata(project_name: str, use_case_type: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive project metadata with proper serialization"""
    
    try:
        from model_manager import ModelCapability
        capabilities = [cap.value for cap in ModelCapability]
    except:
        capabilities = ["general", "coding", "architecture", "documentation"]
    
    # Ensure config_data is serializable by deep cleaning
    def make_serializable(obj):
        """Recursively make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        else:
            try:
                json.dumps(obj)  # Test if serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    serializable_config = make_serializable(config_data)
    
    metadata = {
        "project": {
            "name": project_name,
            "use_case_type": use_case_type,
            "created_at": datetime.now().isoformat(),
            "framework_version": "2.0.0"
        },
        "configuration": serializable_config,
        "generated_by": "Enterprise Multi-Agent SDLC Framework",
        "capabilities": capabilities,
        "quality_metrics": {
            "code_coverage_target": "80%",
            "documentation_completeness": "Complete",
            "security_scan_status": "Required",
            "performance_test_status": "Required"
        }
    }
    
    return metadata

def create_deployment_configs(project_dir: Path, project_name: str, use_case_type: str) -> Dict[str, Path]:
    """Create deployment configuration files"""
    
    config_files = {}
    safe_name = project_name.lower().replace(' ', '-').replace('_', '-')
    
    # Docker Compose
    docker_compose = f"""version: '3.8'

services:
  {safe_name}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/{safe_name.replace('-', '_')}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: {safe_name.replace('-', '_')}
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
"""
    
    compose_file = project_dir / "docker-compose.yml"
    save_file(docker_compose, compose_file, "Docker Compose configuration")
    config_files["docker-compose"] = compose_file
    
    # Basic Dockerfile
    dockerfile_content = f"""FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/app.py"]
"""
    
    dockerfile = project_dir / "Dockerfile"
    save_file(dockerfile_content, dockerfile, "Dockerfile")
    config_files["dockerfile"] = dockerfile
    
    # Kubernetes manifests for enterprise use cases
    if use_case_type in ["text_to_sql", "microservices_platform", "enterprise_web_app"]:
        k8s_dir = project_dir / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        # Deployment manifest
        deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {safe_name}
  labels:
    app: {safe_name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {safe_name}
  template:
    metadata:
      labels:
        app: {safe_name}
    spec:
      containers:
      - name: app
        image: {safe_name}:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: {safe_name}-service
spec:
  selector:
    app: {safe_name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""
        
        deployment_file = k8s_dir / "deployment.yaml"
        save_file(deployment_yaml, deployment_file, "Kubernetes Deployment")
        config_files["k8s-deployment"] = deployment_file
    
    return config_files
