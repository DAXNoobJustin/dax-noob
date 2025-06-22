"""
Interactive Authentication Manager for Power BI/Analysis Services
Provides interactive login similar to DAX Studio and Tabular Editor
"""

import asyncio
import logging
import msal
import webbrowser
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time
import subprocess
import sys
import os

logger = logging.getLogger(__name__)


class InteractiveAuthManager:
    """Handles interactive authentication to Power BI/Analysis Services"""
    
    def __init__(self):
        # Power BI service application (public client)
        self.client_id = "ea0616ba-638b-4df5-95b9-636659ae5121"  # Power BI Service
        self.authority = "https://login.microsoftonline.com/common"
        self.scopes = ["https://analysis.windows.net/powerbi/api/.default"]
        self.redirect_uri = "http://localhost:8400"
        
    async def interactive_login(self, server_url: str) -> str:
        """
        Perform interactive login and return connection string
        """
        try:
            logger.info("Starting interactive authentication...")
            
            # Create MSAL public client application
            app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority
            )
            
            # Try to get token silently first (if user has logged in before)
            accounts = app.get_accounts()
            result = None
            
            if accounts:
                logger.info("Found existing accounts, attempting silent login...")
                result = app.acquire_token_silent(
                    scopes=self.scopes,
                    account=accounts[0]
                )
            
            # If silent login failed, do interactive login
            if not result:
                logger.info("Silent login failed or no accounts found, starting interactive login...")
                
                # Start interactive login flow
                result = app.acquire_token_interactive(
                    scopes=self.scopes,
                    prompt="select_account",  # Allow user to select account
                    timeout=300  # 5 minutes timeout
                )
            
            if "access_token" in result:
                logger.info("Authentication successful!")
                
                # Build connection string
                connection_string = self._build_connection_string(
                    server_url, 
                    result["access_token"]
                )
                
                return connection_string
                
            else:
                error_msg = result.get("error_description", result.get("error", "Unknown error"))
                raise Exception(f"Authentication failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Interactive login failed: {e}")
            raise Exception(f"Interactive login failed: {str(e)}")
    
    def _build_connection_string(self, server_url: str, access_token: str) -> str:
        """Build XMLA connection string with access token"""
        
        # Parse server URL to get the data source
        parsed_url = urlparse(server_url)
        
        if "powerbi.com" in parsed_url.netloc:
            # Power BI XMLA endpoint
            data_source = server_url
        else:
            # Analysis Services server
            data_source = f"{parsed_url.netloc}:{parsed_url.port or 443}"
        
        connection_string = (
            f"Provider=MSOLAP;"
            f"Data Source={data_source};"
            f"User ID=;"
            f"Password={access_token};"
            f"Persist Security Info=True;"
            f"Impersonation Level=Impersonate;"
        )
        
        return connection_string
    
    async def login_with_device_code(self, server_url: str) -> str:
        """
        Alternative login method using device code flow (for environments without browser)
        """
        try:
            logger.info("Starting device code authentication...")
            
            app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority
            )
            
            # Start device flow
            flow = app.initiate_device_flow(scopes=self.scopes)
            
            if "user_code" not in flow:
                raise Exception("Failed to create device flow")
            
            # Show device code to user
            print("\n" + "="*60)
            print("🔐 DEVICE CODE AUTHENTICATION")
            print("="*60)
            print(f"1. Go to: {flow['verification_uri']}")
            print(f"2. Enter code: {flow['user_code']}")
            print("3. Complete the sign-in process")
            print("="*60)
            
            # Try to open browser automatically
            try:
                webbrowser.open(flow["verification_uri"])
                print("✅ Browser opened automatically")
            except:
                print("ℹ️ Please open the URL manually in your browser")
            
            # Poll for completion
            result = app.acquire_token_by_device_flow(flow)
            
            if "access_token" in result:
                logger.info("Device code authentication successful!")
                connection_string = self._build_connection_string(
                    server_url, 
                    result["access_token"]
                )
                return connection_string
            else:
                error_msg = result.get("error_description", result.get("error", "Unknown error"))
                raise Exception(f"Device code authentication failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Device code login failed: {e}")
            raise Exception(f"Device code login failed: {str(e)}")
    
    async def get_cached_token(self) -> Optional[str]:
        """Get cached access token if available"""
        try:
            app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority
            )
            
            accounts = app.get_accounts()
            if accounts:
                result = app.acquire_token_silent(
                    scopes=self.scopes,
                    account=accounts[0]
                )
                
                if "access_token" in result:
                    return result["access_token"]
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get cached token: {e}")
            return None
    
    def clear_cache(self):
        """Clear authentication cache"""
        try:
            app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority
            )
            
            accounts = app.get_accounts()
            for account in accounts:
                app.remove_account(account)
                
            logger.info("Authentication cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        try:
            app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority
            )
            
            accounts = app.get_accounts()
            if accounts:
                account = accounts[0]
                return {
                    "username": account.get("username"),
                    "name": account.get("name"),
                    "home_account_id": account.get("home_account_id")
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get user info: {e}")
            return None


class ServicePrincipalAuthManager:
    """Alternative authentication using Service Principal (for automation)"""
    
    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scopes = ["https://analysis.windows.net/powerbi/api/.default"]
    
    async def login(self, server_url: str) -> str:
        """Login using service principal credentials"""
        try:
            logger.info("Authenticating with service principal...")
            
            app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority
            )
            
            result = app.acquire_token_for_client(scopes=self.scopes)
            
            if "access_token" in result:
                logger.info("Service principal authentication successful!")
                
                # Build connection string
                connection_string = self._build_connection_string(
                    server_url, 
                    result["access_token"]
                )
                
                return connection_string
            else:
                error_msg = result.get("error_description", result.get("error", "Unknown error"))
                raise Exception(f"Service principal authentication failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Service principal login failed: {e}")
            raise Exception(f"Service principal login failed: {str(e)}")
    
    def _build_connection_string(self, server_url: str, access_token: str) -> str:
        """Build XMLA connection string with access token"""
        parsed_url = urlparse(server_url)
        
        if "powerbi.com" in parsed_url.netloc:
            data_source = server_url
        else:
            data_source = f"{parsed_url.netloc}:{parsed_url.port or 443}"
        
        connection_string = (
            f"Provider=MSOLAP;"
            f"Data Source={data_source};"
            f"User ID=app:{self.client_id}@{self.tenant_id};"
            f"Password={access_token};"
            f"Persist Security Info=True;"
        )
        
        return connection_string
