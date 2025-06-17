"""
Setup script for Multi-Agent SDLC Automation Framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Multi-Agent SDLC Automation Framework"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="sdlc-automation-framework",
    version="2.0.0",
    description="Enhanced Multi-Agent Generative AI Framework for SDLC Automation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="SDLC Framework Team",
    author_email="team@sdlc-framework.dev",
    url="https://github.com/your-org/sdlc-automation-framework",
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "sdlc-framework=main:main",
            "sdlc-interactive=main:run_interactive_mode",
            "sdlc-demo=main:run_demo",
        ],
    },
    keywords="ai, automation, sdlc, code-generation, multi-agent, software-development, llm, generative-ai",
    project_urls={
        "Bug Reports": "https://github.com/your-org/sdlc-automation-framework/issues",
        "Source": "https://github.com/your-org/sdlc-automation-framework",
        "Documentation": "https://sdlc-framework.readthedocs.io/",
        "Changelog": "https://github.com/your-org/sdlc-automation-framework/blob/main/CHANGELOG.md",
    },
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=4.0"],
        "gpu": ["torch[cuda]", "accelerate"],
        "examples": ["jupyter", "matplotlib", "pandas"],
    },
)
