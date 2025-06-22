#!/usr/bin/env python3
"""
Basic test script for DAX Optimizer MCP Server functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src" / "mcp"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from dax_optimizer_server import DAXKnowledgeBase, DAXOptimizer, PowerBIConnector
        print("✅ Core classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core classes: {e}")
        return False
    
    try:
        import sqlite3
        import requests
        import pandas as pd
        print("✅ Required dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    return True

def test_knowledge_base():
    """Test knowledge base initialization"""
    print("\nTesting knowledge base...")
    
    try:
        from dax_optimizer_server import DAXKnowledgeBase
        
        # Create temporary knowledge base
        kb = DAXKnowledgeBase(cache_dir="./test_kb")
        print("✅ Knowledge base initialized")
        
        # Test search (should return empty results initially)
        results = kb.search("DISTINCTCOUNT", limit=1)
        print(f"✅ Search functionality works (found {len(results)} results)")
        
        # Clean up
        import shutil
        shutil.rmtree("./test_kb", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\nTesting environment...")
    
    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key found")
    else:
        print("⚠️ OpenAI API key not set (required for optimization)")
    
    # Check for Python version
    if sys.version_info >= (3, 8):
        print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is supported")
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} is too old, need 3.8+")
        return False
    
    return True

def test_basic_dax_cleaning():
    """Test DAX query cleaning functionality"""
    print("\nTesting DAX cleaning...")
    
    try:
        from dax_optimizer_server import clean_dax_query
        
        test_query = "<oii>EVALUATE</oii> {1}"
        cleaned = clean_dax_query(test_query)
        expected = "EVALUATE {1}"
        
        if cleaned == expected:
            print("✅ DAX cleaning works correctly")
            return True
        else:
            print(f"❌ DAX cleaning failed: got '{cleaned}', expected '{expected}'")
            return False
            
    except Exception as e:
        print(f"❌ DAX cleaning test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 DAX Optimizer MCP Server - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_imports,
        test_knowledge_base,
        test_basic_dax_cleaning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The MCP server should work correctly.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python run_server.py --check-config")
        print("3. Start server: python run_server.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
