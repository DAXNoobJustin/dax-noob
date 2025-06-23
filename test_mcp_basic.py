"""
Simple test script for the DAX Optimizer MCP Server
This tests the basic server startup and tool listing functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def test_mcp_server():
    """Test basic MCP server functionality"""
    print("🧪 Testing DAX Optimizer MCP Server...")
    
    try:        # Import the server class
        from mcp.dax_server import DAXOptimizerMCPServer
        
        # Create server instance
        server_instance = DAXOptimizerMCPServer()
        print("✅ Server instance created successfully")
        
        # Test tool listing
        tools = await server_instance.server.list_tools()
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        
        print("\n🎉 Basic MCP server test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
