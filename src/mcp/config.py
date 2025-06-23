"""
Configuration management for DAX Optimizer MCP Server
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MCPConfig:
    """MCP Server configuration"""
    server_name: str = "dax-optimizer-mcp-server"
    server_version: str = "1.0.0"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000
    
    # Authentication Configuration
    power_bi_client_id: str = "ea0616ba-638b-4df5-95b9-636659ae5121"
    authority: str = "https://login.microsoftonline.com/common"
    redirect_uri: str = "http://localhost:8400"
    
    # Knowledge Base Configuration
    kb_cache_dir: Optional[str] = None
    kb_update_interval_hours: int = 24
    max_kb_articles: int = 50
    
    # Performance Configuration
    max_optimization_iterations: int = 3
    query_timeout_seconds: int = 300
    performance_cache_ttl: int = 3600
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", self.openai_temperature))
        self.kb_cache_dir = os.getenv("KB_CACHE_DIR", self.kb_cache_dir)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # Validate required configuration
        if not self.openai_api_key:
            import warnings
            warnings.warn("OpenAI API key not configured - optimization features will be limited")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key) 
            for key in self.__dataclass_fields__.keys()
        }


# Global configuration instance
config = MCPConfig()
