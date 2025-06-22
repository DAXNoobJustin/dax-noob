"""
Enhanced DAX Optimizer MCP Server Launcher
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mcp.server import main

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced DAX Optimizer MCP Server...")
        print("📡 Server will communicate via stdio for VS Code Agent Mode")
        print("🔗 Connect using XMLA endpoint and optimize your DAX measures!")
        print("-" * 60)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Enhanced DAX Optimizer server stopped by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
