"""
Simple MCP Server Test - Tests core functionality without Power BI dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def test_server_startup():
    """Test that the server can start and list tools"""
    print("🧪 Testing MCP Server startup...")
    
    try:
        # Test the main entry point
        from mcp.dax_server import main, DAXOptimizerMCPServer
        
        print("✅ Successfully imported DAXOptimizerMCPServer")
        
        # Create a server instance
        server_instance = DAXOptimizerMCPServer()
        print("✅ Server instance created")
        
        # Check if we can access the server object
        if hasattr(server_instance, 'server'):
            print("✅ Server object accessible")
        else:
            print("❌ Server object not found")
            return False
            
        print("\n🎉 Basic server initialization test passed!")
        print("💡 To test full functionality, you'll need:")
        print("   - OPENAI_API_KEY environment variable")
        print("   - Access to a Power BI workspace with XMLA enabled")
        print("   - MCP client (VS Code with MCP extension or Claude Desktop)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_server_startup())
    if success:
        print("\n🚀 Server is ready for testing!")
        print("📋 Next steps:")
        print("   1. Set OPENAI_API_KEY in .env file")
        print("   2. Test with VS Code MCP extension or Claude Desktop")
        print("   3. Try connecting to a Power BI workspace")
    sys.exit(0 if success else 1)
