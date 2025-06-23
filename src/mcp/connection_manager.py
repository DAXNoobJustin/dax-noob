"""
Connection Manager - Centralized connection state management
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import re
from .config import config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages Analysis Services connections and state"""
    
    def __init__(self):
        self._connection_string: Optional[str] = None
        self._current_database: Optional[str] = None
        self._server_info: Dict[str, Any] = {}
        self._connection_time: Optional[datetime] = None
        self._is_connected: bool = False
    
    @property
    def connection_string(self) -> Optional[str]:
        """Get current connection string"""
        return self._connection_string
    
    @property
    def current_database(self) -> Optional[str]:
        """Get current database name"""
        return self._current_database
    
    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._is_connected and self._connection_string is not None
    
    @property
    def server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return self._server_info.copy()
    
    def set_connection(self, connection_string: str, server_info: Dict[str, Any] = None):
        """Set connection details"""
        self._connection_string = connection_string
        self._server_info = server_info or {}
        self._connection_time = datetime.now()
        self._is_connected = True
        
        # Extract database from connection string if present
        if "Initial Catalog=" in connection_string:
            match = re.search(r'Initial Catalog=([^;]+)', connection_string)
            if match:
                self._current_database = match.group(1)
        
        logger.info(f"Connection established to {self._current_database or 'server'}")
    
    def set_database(self, database_name: str) -> str:
        """Switch to a different database and return updated connection string"""
        if not self._connection_string:
            raise Exception("No active connection")
        
        # Update connection string with new database
        if "Initial Catalog=" in self._connection_string:
            # Replace existing catalog
            updated_connection = re.sub(
                r'Initial Catalog=[^;]*;?',
                f'Initial Catalog={database_name};',
                self._connection_string
            )
        else:
            # Add catalog to connection string
            updated_connection = self._connection_string.rstrip(';') + f";Initial Catalog={database_name};"
        
        self._connection_string = updated_connection
        self._current_database = database_name
        
        logger.info(f"Switched to database: {database_name}")
        return updated_connection
    
    def get_connection_for_component(self) -> str:
        """Get connection string for components"""
        if not self.is_connected:
            raise Exception("No active connection available")
        return self._connection_string
    
    def disconnect(self):
        """Disconnect and clear state"""
        self._connection_string = None
        self._current_database = None
        self._server_info = {}
        self._connection_time = None
        self._is_connected = False
        logger.info("Connection closed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            "is_connected": self._is_connected,
            "current_database": self._current_database,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "server_info": self._server_info,
            "has_openai": config.openai_api_key is not None
        }


# Global connection manager instance
connection_manager = ConnectionManager()
