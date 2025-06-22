#!/usr/bin/env python3
"""
DAX Optimizer MCP Server CLI Runner

This script starts the DAX Optimizer MCP Server for use with VS Code, Claude, or ChatGPT.
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add the src directory to path so we can import our modules
current_dir = Path(__file__).parent
src_dir = current_dir / "src" / "mcp"
sys.path.insert(0, str(src_dir))

from dax_optimizer_server import main

def setup_environment():
    """Setup environment variables from config file if it exists"""
    config_file = current_dir / "config.env"
    if config_file.exists():
        from dotenv import load_dotenv
        load_dotenv(config_file)
        print(f"✅ Loaded configuration from {config_file}")
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your environment or in the config.env file.")
        print("Example: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    return True

def main_cli():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DAX Optimizer MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                    # Start the MCP server
  python run_server.py --check-config    # Check configuration
  python run_server.py --help            # Show this help

For VS Code:
  1. Start this server: python run_server.py
  2. In VS Code, use Copilot Agent Mode to connect to MCP servers
  3. Use tools like 'optimize_measure', 'search_dax_knowledge', etc.

For Claude/ChatGPT:
  1. Start this server: python run_server.py  
  2. Configure your client to connect to this MCP server
  3. Use the DAX optimization tools in your conversations
        """
    )
    
    parser.add_argument(
        "--check-config", 
        action="store_true",
        help="Check configuration and exit"
    )
    
    parser.add_argument(
        "--version", 
        action="store_true",
        help="Show version information"
    )
    
    args = parser.parse_args()
    
    if args.version:
        print("DAX Optimizer MCP Server v1.0.0")
        print("Built for optimizing DAX measures and queries")
        return
    
    print("🚀 DAX Optimizer MCP Server")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    if args.check_config:
        print("✅ Configuration check passed!")
        print("\nOptional: Update knowledge base before first use:")
        print("  Use the 'update_knowledge_base' tool after connecting")
        return
    
    print("Starting MCP server...")
    print("💡 Tip: Use 'connect_powerbi' tool first to establish connection")
    print("📚 Use 'update_knowledge_base' to download DAX optimization knowledge")
    print("🎯 Use 'optimize_measure' to optimize your DAX measures")
    print("\nServer running... (Press Ctrl+C to stop)")
    
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_cli()
