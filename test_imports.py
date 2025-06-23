"""
Minimal DAX Optimizer MCP Server Test
Tests if the server can start without full dependencies
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if core modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test MCP package
        import mcp
        print("✓ mcp package imported")
        
        # Test if our modules can be imported
        from mcp.auth_manager import InteractiveAuthManager
        print("✓ auth_manager imported")
        
        from mcp.dax_analyzer import DAXAnalyzer  
        print("✓ dax_analyzer imported")
        
        from mcp.knowledge_base import DAXKnowledgeBase
        print("✓ knowledge_base imported")
        
        print("\n🎉 All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 DAX Optimizer MCP Server - Import Test\n")
    success = test_imports()
    
    if success:
        print("\n✓ Server modules are ready!")
        print("💡 Next: Set up OPENAI_API_KEY and test with MCP client")
    else:
        print("\n✗ Setup issues detected")
        print("💡 Check dependencies and file structure")
        
    sys.exit(0 if success else 1)
