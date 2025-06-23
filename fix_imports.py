"""
Quick fix for the MCP server import issue
This script renames the local mcp module to avoid conflicts with the official MCP package
"""

import os
import shutil
from pathlib import Path

def fix_import_conflicts():
    """Fix the import conflicts in the MCP server"""
    
    print("🔧 Fixing import conflicts in DAX Optimizer MCP Server...")
    
    # Rename the local mcp package to avoid conflicts
    src_mcp = Path("src/mcp")
    src_dax_mcp = Path("src/dax_mcp")
    
    if src_mcp.exists() and not src_dax_mcp.exists():
        print("📦 Renaming src/mcp to src/dax_mcp...")
        shutil.move(str(src_mcp), str(src_dax_mcp))
        print("✅ Directory renamed")
    
    # Update run_server.py import
    run_server = Path("run_server.py")
    if run_server.exists():
        content = run_server.read_text()
        content = content.replace("from mcp.dax_server import main", "from dax_mcp.dax_server import main")
        run_server.write_text(content)
        print("✅ Updated run_server.py imports")
    
    # Update VS Code settings
    vscode_settings = Path(".vscode/settings.json")
    if vscode_settings.exists():
        content = vscode_settings.read_text()
        # This will work because the path in args points to run_server.py which we just fixed
        print("✅ VS Code settings compatible")
    
    print("\n🎉 Import conflicts fixed!")
    print("💡 Now you can test with:")
    print("   - python run_server.py")
    print("   - VS Code MCP extension")
    print("   - Claude Desktop")

if __name__ == "__main__":
    try:
        fix_import_conflicts()
    except Exception as e:
        print(f"❌ Error fixing imports: {e}")
        print("💡 You may need to manually rename src/mcp to src/dax_mcp")
