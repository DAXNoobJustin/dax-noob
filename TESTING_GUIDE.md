# DAX Optimizer MCP Server - Testing Guide

## Overview
Your DAX Optimizer MCP Server is a sophisticated tool for optimizing DAX measures in Power BI and Analysis Services. Here's how to test it effectively.

## Current Status
✅ **Project Structure**: Well organized with proper MCP server implementation
✅ **Dependencies**: Core packages are installed (mcp, openai, pandas, etc.)
✅ **Configuration**: VS Code and Claude Desktop configs are ready
⚠️  **Import Issue**: There's a module naming conflict that needs to be resolved

## Quick Testing Options

### 1. VS Code with MCP Extension (Recommended)
This is the easiest way to test your MCP server:

**Setup:**
1. Install the "Model Context Protocol" extension in VS Code
2. Open your project in VS Code
3. The MCP server is already configured in `.vscode/settings.json`
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

**Testing:**
- Open VS Code chat/assistant
- Your DAX optimizer tools should be available
- Try commands like "list available databases" or "help me optimize a DAX measure"

### 2. Claude Desktop Integration
**Setup:**
1. Copy the contents of `claude_mcp_config.json` to Claude Desktop's MCP settings
2. Update the path to match your system
3. Add your OpenAI API key to the configuration

**Testing:**
- Restart Claude Desktop
- Ask Claude to "list available MCP tools"
- Test with: "Connect to my Power BI workspace"

### 3. Command Line Testing
**Direct Server Test:**
```powershell
cd "c:\Users\justinmartin\Projects\dax-noob\dax-noob"
python run_server.py
```

## Available Tools for Testing

Once connected, your MCP server provides these tools:

| Tool | Purpose | Test Command |
|------|---------|--------------|
| `interactive_login` | Connect to Power BI | "Login to powerbi://api.powerbi.com/v1.0/myorg/MyWorkspace" |
| `list_databases` | Show available datasets | "List my Power BI datasets" |
| `optimize_dax_measure` | AI-powered optimization | "Optimize this DAX: SUM(Sales[Amount])" |
| `analyze_dax_performance` | Performance analysis | "Analyze the performance of my Total Sales measure" |
| `search_dax_knowledge` | Knowledge base search | "Search for DAX optimization best practices" |
| `get_model_metadata` | Model information | "Show me information about my data model" |

## Prerequisites for Full Testing

### Power BI Setup
1. **Power BI License**: Pro or Premium Per User (PPU)
2. **XMLA Endpoint**: Enabled in your Power BI workspace settings
3. **Workspace Access**: Admin or Member role in the workspace

### Authentication
- The tool uses interactive authentication (like DAX Studio)
- You'll be prompted to sign in to your Microsoft account
- No need to create app registrations for basic testing

### API Keys
- **OpenAI API Key**: Required for AI-powered optimization features
- Add to `.env` file: `OPENAI_API_KEY=your_key_here`

## Sample Testing Workflow

1. **Start Simple**:
   ```
   "List available MCP tools"
   "Connect to my Power BI workspace"
   ```

2. **Basic Connection**:
   ```
   interactive_login with server_url: "powerbi://api.powerbi.com/v1.0/myorg/YourWorkspaceName"
   ```

3. **Explore Your Data**:
   ```
   "List my databases"
   "Connect to [YourDatasetName]"
   "Show me model metadata"
   ```

4. **Test Optimization**:
   ```
   "Optimize this DAX measure: SUM(Sales[Amount])"
   "Analyze performance of Total Revenue measure"
   ```

## Troubleshooting

### Common Issues:
1. **Import Errors**: Module naming conflict with MCP package
   - **Solution**: The server files need import path fixes
   
2. **Authentication Failed**: 
   - **Solution**: Check XMLA endpoint URL format
   - **Format**: `powerbi://api.powerbi.com/v1.0/myorg/WorkspaceName`

3. **No OpenAI Response**:
   - **Solution**: Verify OPENAI_API_KEY in `.env` file

### Quick Fixes:
```bash
# Check Python environment
python --version

# Verify MCP package
python -c "import mcp; print('MCP OK')"

# Test OpenAI connection
python -c "import openai; print('OpenAI OK')"
```

## Next Steps for Development

1. **Fix Import Issue**: Resolve the module naming conflict
2. **Add Unit Tests**: Create proper pytest test suite
3. **Mock Testing**: Add tests that don't require Power BI connection
4. **Documentation**: Add more detailed API documentation

## Support

- **Power BI XMLA**: [Microsoft Documentation](https://docs.microsoft.com/en-us/power-bi/admin/service-premium-connect-tools)
- **MCP Protocol**: [Model Context Protocol Docs](https://modelcontextprotocol.io)
- **DAX Optimization**: Your knowledge base includes kb.daxoptimizer.com resources

---

**Ready to Test?** Start with VS Code + MCP extension for the best experience!
