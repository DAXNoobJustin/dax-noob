# DAX Optimizer MCP Server

A powerful Model Context Protocol (MCP) server designed to optimize DAX measures and queries for Power BI. This tool helps you improve the performance of your DAX code while ensuring results remain identical.

## 🎯 Features

- **DAX Measure Optimization**: Automatically generates and tests optimized variants of your DAX measures
- **Performance Analysis**: Provides detailed timing analysis for DAX queries
- **Knowledge Base Search**: Access optimization patterns from kb.daxoptimizer.com
- **Result Verification**: Ensures optimized measures return identical results to originals  
- **File Context Support**: Upload files (PDFs, docs) to provide additional context for optimization
- **Variant Comparison**: Compare multiple DAX measure implementations side-by-side

## 🚀 Quick Start

### Prerequisites

1. **Power BI Premium Workspace**: You need access to a Premium workspace with XMLA endpoints enabled
2. **Service Principal**: Set up a service principal with Power BI access (or use interactive login)
3. **OpenAI API Key**: Required for AI-powered optimization suggestions
4. **Python 3.8+**: Required for running the MCP server

### Installation

1. **Clone/Download the project**:
   ```bash
   git clone <repository-url>
   cd dax-noob
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file or set environment variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Test configuration**:
   ```bash
   python run_server.py --check-config
   ```

### Running the Server

**For VS Code (Recommended)**:
1. Open the project in VS Code
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
