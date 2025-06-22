# DAX Optimizer MCP Server

A powerful Model Context Protocol (MCP) server designed to optimize DAX measures and queries for Power BI and Analysis Services. This tool provides interactive authentication, comprehensive performance analysis, and AI-powered optimization suggestions without requiring Microsoft Fabric dependencies.

## 🎯 Key Features

- **🔐 Interactive Authentication**: Login like DAX Studio/Tabular Editor with account selection
- **⚡ DAX Measure Optimization**: AI-powered optimization with performance testing
- **📊 Performance Analysis**: Detailed query timing and resource usage analysis
- **🔍 Knowledge Base Search**: Access optimization patterns from kb.daxoptimizer.com
- **✅ Result Verification**: Ensures optimized measures return identical results
- **📁 File Context Support**: Upload documents to provide additional optimization context
- **⚖️ Variant Comparison**: Compare multiple DAX implementations side-by-side
- **🌐 No Fabric Dependencies**: Works with any Power BI Pro/PPU license via XMLA

## 🚀 Quick Start

### Prerequisites

1. **Power BI Access**: Pro or Premium Per User (PPU) license
2. **XMLA Endpoints**: Must be enabled in your Power BI workspace
3. **Python 3.8+**: For running the MCP server
4. **OpenAI API Key**: For AI-powered optimization features
5. **VS Code**: With MCP extension for best experience

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dax-noob
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Configure VS Code MCP**:
   The project includes a VS Code configuration. Make sure you have the MCP extension installed.

### Running the Server

**For VS Code (Recommended)**:
1. Open the project in VS Code
2. The MCP server should be automatically configured
3. Use the MCP tools in VS Code to interact with Power BI

**For Command Line**:
```bash
python run_server.py
```

**For Claude Desktop**:
Add this configuration to your Claude Desktop MCP settings:
```json
{
  "mcpServers": {
    "dax-optimizer": {
      "command": "python",
      "args": ["c:\\path\\to\\your\\project\\run_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 🔧 Usage Guide

### 1. Authentication and Connection

**Interactive Login** (Recommended):
```
Use the 'interactive_login' tool with your Power BI XMLA endpoint:
- Server URL: powerbi://api.powerbi.com/v1.0/myorg/YourWorkspaceName
```

**Connection String** (Alternative):
```
Use 'connect_with_connection_string' if you have a custom connection string
```

### 2. Basic Workflow

1. **Connect**: Use `interactive_login` with your workspace XMLA endpoint
2. **List Databases**: Use `list_databases` to see available datasets  
3. **Connect to Dataset**: Use `connect_to_database` with your dataset name
4. **Get Model Info**: Use `get_model_metadata` to understand your model structure
5. **Optimize Measures**: Use `optimize_dax_measure` to improve performance

### 3. Available Tools

| Tool | Description |
|------|-------------|
| `interactive_login` | Login with account selection (like DAX Studio) |
| `list_databases` | Show available datasets in your workspace |
| `connect_to_database` | Connect to a specific dataset |
| `get_model_metadata` | Get comprehensive model information |
| `optimize_dax_measure` | AI-powered measure optimization |
| `analyze_dax_performance` | Detailed performance analysis |
| `compare_dax_variants` | Compare multiple DAX implementations |
| `search_dax_knowledge` | Search optimization knowledge base |
| `upload_context_file` | Add files for optimization context |
| `get_server_info` | Get Analysis Services server details |

### 4. Example: Optimizing a Measure

```
1. interactive_login: 
   server_url: "powerbi://api.powerbi.com/v1.0/myorg/MyWorkspace"

2. list_databases (to see available datasets)

3. connect_to_database:
   database_name: "My Sales Dataset"

4. optimize_dax_measure:
   measure_name: "Total Sales"
   dax_expression: "SUM(Sales[Amount])"
   max_iterations: 3
```

## 🔍 Architecture

### Core Components

- **Auth Manager**: Handles interactive authentication using MSAL
- **DAX Analyzer**: Core optimization engine with OpenAI integration
- **Performance Analyzer**: Query timing and resource usage analysis
- **Model Metadata Extractor**: Comprehensive model information via DMV queries
- **Knowledge Base**: Scrapes and searches DAX optimization resources

### Key Technologies

- **XMLA/DMV Queries**: Direct Analysis Services communication
- **MSAL**: Microsoft Authentication Library for interactive login
- **OpenAI GPT**: AI-powered optimization suggestions
- **SQLite FTS**: Full-text search for knowledge base
- **BeautifulSoup**: Web scraping for optimization resources

## 🛠️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
KB_CACHE_DIR=./kb_cache
KB_UPDATE_INTERVAL_HOURS=24
DEFAULT_MAX_ITERATIONS=3
OPTIMIZATION_TIMEOUT_SECONDS=300
```

### VS Code Settings

The project includes a `.vscode/settings.json` file with MCP configuration. Ensure your paths are correct for your environment.

### Authentication Options

1. **Interactive Login** (Primary): Browser-based authentication with account selection
2. **Device Code Flow**: For environments without browser access
3. **Service Principal**: For automation scenarios

## 📊 Performance Thresholds

- **Excellent**: < 100ms
- **Good**: 100-500ms  
- **Moderate**: 500ms-2s
- **Slow**: 2s-10s
- **Very Slow**: > 10s

## 🤝 Contributing

This is an open-source project! Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license here]

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: Requires Python 3.8+

**Authentication Failures**:
- Verify XMLA endpoints are enabled in Power BI admin portal
- Check workspace access permissions
- Ensure correct XMLA endpoint format

**Connection Issues**:
- Verify network connectivity to Power BI service
- Check firewall settings
- Validate workspace name in XMLA endpoint

**Performance Issues**:
- Consider using device code flow in restricted environments
- Check OpenAI API rate limits
- Monitor Power BI capacity usage

### Getting Help

1. Check the troubleshooting section above
2. Review VS Code MCP extension documentation
3. Open an issue on GitHub with detailed error information
4. Include relevant log output from the server

## 🚀 Roadmap

- [ ] DAX Studio library integration
- [ ] Tabular Editor integration
- [ ] Advanced query plan analysis
- [ ] Batch optimization capabilities
- [ ] Custom optimization rules engine
- [ ] Performance benchmarking suite
- [ ] Integration with CI/CD pipelines
2. Use Ctrl+Shift+P → "Tasks: Run Task" → "Start DAX Optimizer MCP Server"
3. The server will start and be available for VS Code Copilot Agent Mode

**For Command Line**:
```bash
python run_server.py
```

**For Claude/ChatGPT**:
Configure your MCP client to connect to this server (see client-specific documentation).

## 🛠️ Available Tools

### Core Tools

#### `connect_powerbi`
Connect to your Power BI dataset via XMLA endpoint.

**Parameters**:
- `xmla_endpoint`: Power BI XMLA endpoint (e.g., "powerbi://api.powerbi.com/v1.0/myorg/YourWorkspace")  
- `tenant_id`: Azure AD tenant ID
- `client_id`: Service principal application ID
- `client_secret`: Service principal secret
- `initial_catalog`: Dataset/semantic model name

**Example**:
```json
{
  "xmla_endpoint": "powerbi://api.powerbi.com/v1.0/myorg/HelixFabric-Insights",
  "tenant_id": "your-tenant-id",
  "client_id": "your-client-id", 
  "client_secret": "your-client-secret",
  "initial_catalog": "Azure Data Insights - OneLake"
}
```

#### `optimize_measure`
Optimize a DAX measure for better performance.

**Parameters**:
- `measure_name`: Name of the measure to optimize
- `dax_expression`: Current DAX expression  
- `max_iterations`: Maximum optimization iterations (default: 3)

**Example**:
```json
{
  "measure_name": "Total Sales",
  "dax_expression": "SUM(Sales[Amount])",
  "max_iterations": 3
}
```

#### `analyze_query_performance`
Analyze the performance of a DAX query.

**Parameters**:
- `dax_query`: DAX query to analyze

#### `compare_measure_variants`
Compare multiple DAX measure implementations.

**Parameters**:
- `measure_name`: Name of the measure
- `variants`: Array of variants with `name` and `dax` properties

### Knowledge Base Tools

#### `search_dax_knowledge`
Search the optimization knowledge base.

**Parameters**:
- `query`: Search query
- `limit`: Maximum results (default: 5)

#### `update_knowledge_base`
Update the local knowledge base from kb.daxoptimizer.com.

### Context Tools

#### `upload_file_context`
Upload a file to provide context for optimization.

**Parameters**:
- `filename`: Name of the file
- `content`: File content

## 📊 Usage Examples

### Basic Optimization Workflow

1. **Connect to Power BI**:
   ```
   Use connect_powerbi with your XMLA endpoint details
   ```

2. **Update Knowledge Base** (first time):
   ```
   Use update_knowledge_base to download optimization patterns
   ```

3. **Optimize a Measure**:
   ```
   Use optimize_measure with your DAX expression
   ```

4. **Search for Patterns**:
   ```
   Use search_dax_knowledge to find relevant optimization techniques
   ```

### Advanced Workflows

**Upload Context Files**:
Upload DAX best practices documents or model documentation to provide additional context for optimization suggestions.

**Compare Variants**:
Test multiple implementations of the same measure to find the fastest one.

**Performance Analysis**:
Analyze complex queries to identify performance bottlenecks.

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for AI-powered optimization
- `KB_CACHE_DIR`: Directory for knowledge base cache (default: ./kb_cache)
- `DEFAULT_MAX_ITERATIONS`: Default optimization iterations (default: 3)

### Performance Thresholds

The system categorizes query performance as:
- **Excellent**: < 100ms ⚡
- **Good**: 100-500ms ✅  
- **Moderate**: 500ms-2s ⚠️
- **Slow**: > 2s 🐌 (optimization recommended)

## 🏗️ Architecture

### Core Components

1. **DAXKnowledgeBase**: Manages optimization patterns from kb.daxoptimizer.com
2. **PowerBIConnector**: Handles XMLA connections and query execution  
3. **DAXOptimizer**: Orchestrates the optimization process
4. **DAXOptimizerMCPServer**: MCP server implementation

### Optimization Process

1. **Baseline Measurement**: Execute original DAX and measure performance
2. **Context Gathering**: Collect relevant optimization patterns from knowledge base
3. **Variant Generation**: Use AI to generate optimized DAX variants
4. **Testing & Verification**: Execute variants and verify results match baseline
5. **Best Selection**: Identify fastest variant that produces identical results

## 🔍 Troubleshooting

### Common Issues

**Connection Failed**:
- Verify XMLA endpoint URL is correct
- Check service principal permissions
- Ensure Premium workspace and XMLA endpoints are enabled

**Optimization Not Working**:
- Verify OpenAI API key is set
- Check if knowledge base is populated (`update_knowledge_base`)
- Ensure DAX syntax is valid

**Performance Issues**:
- Large datasets may cause longer execution times
- Consider using smaller sample queries for testing
- Check network connectivity to Power BI service

### Debug Mode

Run with VS Code debugger for detailed logging:
1. Set breakpoints in the code
2. Use "Debug DAX Optimizer MCP Server" launch configuration
3. Monitor logs in the integrated terminal

## 🚦 VS Code Integration

### Setting Up Agent Mode

1. Ensure VS Code has the latest Copilot extension
2. Enable Agent Mode in VS Code settings
3. Start the MCP server using the provided VS Code task
4. Use Copilot commands like:
   - "Optimize this DAX measure"
   - "Search for DISTINCTCOUNT optimization patterns" 
   - "Compare these measure variants"

### Available VS Code Tasks

- **Start DAX Optimizer MCP Server**: Starts the server in background
- **Check MCP Server Configuration**: Validates setup
- **Install Dependencies**: Installs required Python packages
- **Update DAX Knowledge Base**: Downloads latest optimization patterns

## 🤝 Contributing

This project is built to be extensible. Areas for contribution:

- Additional optimization patterns and heuristics
- Enhanced server timing collection (DAX Studio integration)
- Support for other semantic model platforms
- UI for easier configuration and monitoring

## 📝 License

See LICENSE file for details.

## 🔗 Related Resources

- [DAX Patterns](https://www.daxpatterns.com/)
- [SQLBI Optimization Articles](https://sqlbi.com/)
- [DAX Optimizer Knowledge Base](https://kb.daxoptimizer.com/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [VS Code Agent Mode Documentation](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode)
