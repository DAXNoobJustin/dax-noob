"""
DAX Optimizer MCP Server Package
Provides comprehensive DAX optimization capabilities for Power BI and Analysis Services
"""

__version__ = "1.0.0"
__author__ = "DAX Optimization Team"

from .dax_server import DAXOptimizerMCPServer, main
from .dax_analyzer import DAXAnalyzer
from .auth_manager import InteractiveAuthManager, ServicePrincipalAuthManager
from .knowledge_base import DAXKnowledgeBase
from .model_metadata import ModelMetadataExtractor
from .performance_analyzer import PerformanceAnalyzer
from .query_validator import DAXQueryValidator
from .connection_manager import ConnectionManager
from .config import config

__all__ = [
    "DAXOptimizerMCPServer",
    "main",
    "DAXAnalyzer", 
    "InteractiveAuthManager",
    "ServicePrincipalAuthManager",
    "DAXKnowledgeBase",
    "ModelMetadataExtractor",
    "PerformanceAnalyzer",
    "DAXQueryValidator",
    "ConnectionManager",
    "config"
]
