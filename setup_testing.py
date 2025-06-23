"""
DAX Optimizer MCP Server Testing Guide and Setup
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('mcp', 'mcp'), 
        ('python-dotenv', 'dotenv'), 
        ('openai', 'openai'), 
        ('pandas', 'pandas'),
        ('requests', 'requests'), 
        ('beautifulsoup4', 'bs4'), 
        ('msal', 'msal')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing.append(package_name)
    
    return missing

def setup_environment():
    """Setup the testing environment"""
    print("🔧 Setting up DAX Optimizer MCP Server testing environment...\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n📦 Checking dependencies:")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("💡 Install them with: pip install " + " ".join(missing))
        return False
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("\n⚠️  .env file not found")
        print("💡 Create .env file with:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        
        # Create example .env
        with open(".env.example", "w") as f:
            f.write("# DAX Optimizer MCP Server Environment Variables\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("# Optional: Power BI tenant settings\n")
            f.write("# TENANT_ID=your_tenant_id\n")
            f.write("# CLIENT_ID=your_client_id\n")
        print("📝 Created .env.example file for reference")
    else:
        print("✅ .env file found")
    
    return True

def test_mcp_integration():
    """Test MCP integration options"""
    print("\n🧪 MCP Integration Test Options:\n")
    
    print("1. 📱 VS Code with MCP Extension:")
    print("   - Install 'Model Context Protocol' extension in VS Code")
    print("   - Open this project in VS Code")
    print("   - MCP server should auto-configure from .vscode/settings.json")
    print("   - Use MCP tools directly in VS Code chat\n")
    
    print("2. 💬 Claude Desktop:")
    print("   - Add configuration to Claude Desktop MCP settings")
    print("   - Use the claude_mcp_config.json file created in this directory")
    print("   - Copy contents to Claude's MCP configuration\n")
    
    print("3. 🖥️  Command Line Testing:")
    print("   - Run: python run_server.py")
    print("   - Server will start in stdio mode")
    print("   - Communicate via JSON-RPC over stdin/stdout\n")

def create_test_scripts():
    """Create simple test scripts"""
    
    # Create a minimal server test
    test_content = '''"""
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
        print("✅ mcp package imported")
        
        # Test if our modules can be imported
        from mcp.auth_manager import InteractiveAuthManager
        print("✅ auth_manager imported")
        
        from mcp.dax_analyzer import DAXAnalyzer  
        print("✅ dax_analyzer imported")
        
        from mcp.knowledge_base import DAXKnowledgeBase
        print("✅ knowledge_base imported")
        
        print("\\n🎉 All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 DAX Optimizer MCP Server - Import Test\\n")
    success = test_imports()
    
    if success:
        print("\\n✅ Server modules are ready!")
        print("💡 Next: Set up OPENAI_API_KEY and test with MCP client")
    else:
        print("\\n❌ Setup issues detected")
        print("💡 Check dependencies and file structure")
        
    sys.exit(0 if success else 1)
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_content)
    print("📝 Created test_imports.py")

def main():
    """Main setup and test guide"""
    print("🚀 DAX Optimizer MCP Server - Testing Setup\n")
    print("=" * 60)
    
    if not setup_environment():
        print("\n❌ Environment setup failed. Please fix issues above.")
        return
    
    print("\n✅ Environment setup complete!")
    
    create_test_scripts()
    test_mcp_integration()
    
    print("🎯 Quick Test Steps:")
    print("1. Set OPENAI_API_KEY in .env file")
    print("2. Run: python test_imports.py")
    print("3. Choose an MCP client (VS Code recommended)")
    print("4. Test basic tools like 'interactive_login'")
    print("5. Connect to a Power BI workspace with XMLA enabled")
    
    print("\n📚 For full testing, you'll need:")
    print("   - Power BI Pro/PPU license")
    print("   - Workspace with XMLA endpoint enabled")
    print("   - OpenAI API key for AI-powered features")
    
    print("\n🎉 Ready to test your MCP DAX Optimizer!")

if __name__ == "__main__":
    main()
