"""
DAX Optimizer MCP Server - Clean Implementation
Optimizes DAX measures with interactive authentication and DMV-based analysis
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp.types import Tool, TextContent, Resource, Prompt

# Core functionality imports
try:
    from .auth_manager import InteractiveAuthManager
    from .dax_analyzer import DAXAnalyzer
    from .knowledge_base import DAXKnowledgeBase
    from .model_metadata import ModelMetadataExtractor
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from auth_manager import InteractiveAuthManager
    from dax_analyzer import DAXAnalyzer
    from knowledge_base import DAXKnowledgeBase
    from model_metadata import ModelMetadataExtractor
    from performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class DAXOptimizerMCPServer:
    """Main MCP Server for DAX Optimization"""
    
    def __init__(self):
        self.server = Server("dax-optimizer-mcp-server")
        self.auth_manager = InteractiveAuthManager()
        self.dax_analyzer = None
        self.kb = DAXKnowledgeBase()
        self.metadata_extractor = None
        self.performance_analyzer = None
        self.is_connected = False
        self.connection_info = {}
        self.uploaded_files = {}
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP tool handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="interactive_login",
                    description="Interactive login to Power BI/Analysis Services (like DAX Studio)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "server_url": {
                                "type": "string", 
                                "description": "XMLA endpoint URL (e.g., powerbi://api.powerbi.com/v1.0/myorg/WorkspaceName)"
                            }
                        },
                        "required": ["server_url"]
                    }
                ),
                Tool(
                    name="connect_with_connection_string",
                    description="Connect using a custom connection string",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "connection_string": {
                                "type": "string",
                                "description": "Full XMLA connection string"
                            }
                        },
                        "required": ["connection_string"]
                    }
                ),
                Tool(
                    name="list_databases",
                    description="List available databases/datasets in the connected server",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="connect_to_database",
                    description="Connect to a specific database/dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database_name": {
                                "type": "string",
                                "description": "Name of the database/dataset to connect to"
                            }
                        },
                        "required": ["database_name"]
                    }
                ),                Tool(
                    name="get_model_metadata",
                    description="Get comprehensive model metadata (tables, columns, relationships). Optionally provide a DAX query to get focused metadata based on query dependencies.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dax_query": {
                                "type": "string", 
                                "description": "Optional DAX query to analyze dependencies and return focused metadata"
                            }
                        }
                    }
                ),                Tool(
                    name="define_dax_measures",
                    description="Ensure all measures referenced in a DAX query are fully defined with DEFINE MEASURE statements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dax_query": {"type": "string", "description": "DAX query to analyze and define measures for"}
                        },
                        "required": ["dax_query"]
                    }
                ),
                Tool(
                    name="optimize_dax_measure",
                    description="Optimize a DAX measure with performance analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "measure_name": {"type": "string", "description": "Name of the measure"},
                            "dax_expression": {"type": "string", "description": "Current DAX expression"},
                            "max_iterations": {"type": "integer", "description": "Max optimization attempts", "default": 3}
                        },
                        "required": ["measure_name", "dax_expression"]
                    }
                ),
                Tool(
                    name="analyze_dax_performance",
                    description="Analyze DAX query performance with detailed metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dax_query": {"type": "string", "description": "DAX query to analyze"}
                        },
                        "required": ["dax_query"]
                    }
                ),
                Tool(
                    name="search_dax_knowledge",
                    description="Search DAX optimization knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="update_knowledge_base",
                    description="Update the DAX optimization knowledge base from online sources",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="upload_context_file",
                    description="Upload a file to provide context for optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "File name"},
                            "content": {"type": "string", "description": "File content"}
                        },
                        "required": ["filename", "content"]
                    }
                ),
                Tool(
                    name="compare_dax_variants",
                    description="Compare multiple DAX expressions for performance",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "measure_name": {"type": "string", "description": "Measure name"},
                            "variants": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "dax": {"type": "string"}
                                    },
                                    "required": ["name", "dax"]
                                }
                            }
                        },
                        "required": ["measure_name", "variants"]
                    }
                ),
                Tool(
                    name="get_server_info",
                    description="Get Analysis Services server information and capabilities",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return []
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            return []
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
            try:
                logger.info(f"Handling tool call: {name}")
                
                if name == "interactive_login":
                    result = await self._handle_interactive_login(arguments)
                elif name == "connect_with_connection_string":
                    result = await self._handle_connection_string(arguments)
                elif name == "list_databases":
                    result = await self._handle_list_databases(arguments)
                elif name == "connect_to_database":
                    result = await self._handle_connect_database(arguments)
                elif name == "get_model_metadata":
                    result = await self._handle_get_metadata(arguments)
                elif name == "define_dax_measures":
                    result = await self._handle_define_measures(arguments)
                elif name == "optimize_dax_measure":
                    result = await self._handle_optimize_measure(arguments)
                elif name == "analyze_dax_performance":
                    result = await self._handle_analyze_performance(arguments)
                elif name == "search_dax_knowledge":
                    result = await self._handle_search_knowledge(arguments)
                elif name == "update_knowledge_base":
                    result = await self._handle_update_knowledge_base(arguments)
                elif name == "upload_context_file":
                    result = await self._handle_upload_file(arguments)
                elif name == "compare_dax_variants":
                    result = await self._handle_compare_variants(arguments)
                elif name == "get_server_info":
                    result = await self._handle_get_server_info(arguments)
                else:
                    logger.warning(f"Unknown tool: {name}")
                    return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
                
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Error executing {name}: {str(e)}", exc_info=True)
                return [TextContent(type="text", text=f"❌ Error executing {name}: {str(e)}")]
    
    async def _handle_interactive_login(self, arguments: Dict[str, Any]) -> str:
        """Handle interactive login to Power BI/Analysis Services"""
        server_url = arguments.get("server_url")
        
        try:
            # Attempt interactive login
            connection_string = await self.auth_manager.interactive_login(server_url)
            
            # Initialize components with the connection
            self.dax_analyzer = DAXAnalyzer(connection_string)
            self.metadata_extractor = ModelMetadataExtractor(connection_string)
            self.performance_analyzer = PerformanceAnalyzer(connection_string)
            
            # Test connection
            server_info = await self.dax_analyzer.test_connection()
            
            self.is_connected = True
            self.connection_info = {
                "server_url": server_url,
                "connection_time": datetime.now().isoformat(),
                "server_info": server_info
            }
            
            return f"✅ Successfully connected to Analysis Services!\n\n" \
                   f"🔗 Server: {server_url}\n" \
                   f"📊 Server Info: {server_info.get('version', 'Unknown')}\n" \
                   f"⏰ Connected at: {self.connection_info['connection_time']}\n\n" \
                   f"💡 Next steps:\n" \
                   f"1. Use 'list_databases' to see available datasets\n" \
                   f"2. Use 'connect_to_database' to select a specific dataset\n" \
                   f"3. Start optimizing with 'optimize_dax_measure'"
            
        except Exception as e:
            logger.error(f"Interactive login failed: {e}")
            return f"❌ Interactive login failed: {str(e)}\n\n" \
                   f"💡 Troubleshooting tips:\n" \
                   f"- Ensure you have access to the Power BI workspace\n" \
                   f"- Check that XMLA endpoints are enabled\n" \
                   f"- Verify the server URL format: powerbi://api.powerbi.com/v1.0/myorg/WorkspaceName"
    
    async def _handle_connection_string(self, arguments: Dict[str, Any]) -> str:
        """Handle connection using custom connection string"""
        connection_string = arguments.get("connection_string")
        
        try:
            # Initialize components
            self.dax_analyzer = DAXAnalyzer(connection_string)
            self.metadata_extractor = ModelMetadataExtractor(connection_string)
            self.performance_analyzer = PerformanceAnalyzer(connection_string)
            
            # Test connection
            server_info = await self.dax_analyzer.test_connection()
            
            self.is_connected = True
            self.connection_info = {
                "connection_string": connection_string,
                "connection_time": datetime.now().isoformat(),
                "server_info": server_info
            }
            
            return f"✅ Successfully connected using custom connection string!\n\n" \
                   f"📊 Server Info: {server_info.get('version', 'Unknown')}\n" \
                   f"⏰ Connected at: {self.connection_info['connection_time']}\n\n" \
                   f"💡 Use 'list_databases' to see available datasets"
            
        except Exception as e:
            logger.error(f"Connection string login failed: {e}")
            return f"❌ Connection failed: {str(e)}"
    
    async def _handle_list_databases(self, arguments: Dict[str, Any]) -> str:
        """List available databases"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        try:
            databases = await self.dax_analyzer.list_databases()
            
            result = "📊 Available Databases/Datasets:\n\n"
            for i, db in enumerate(databases, 1):
                result += f"{i}. 📁 {db['name']}\n"
                if db.get('description'):
                    result += f"   📝 {db['description']}\n"
                result += f"   🔄 Last Updated: {db.get('last_update', 'Unknown')}\n\n"
            
            result += "💡 Use 'connect_to_database' with the database name to connect"
            return result
            
        except Exception as e:
            logger.error(f"List databases failed: {e}")
            return f"❌ Failed to list databases: {str(e)}"
    
    async def _handle_connect_database(self, arguments: Dict[str, Any]) -> str:
        """Connect to a specific database"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        database_name = arguments.get("database_name")
        
        try:
            # Update connection to use specific database            await self.dax_analyzer.connect_to_database(database_name)
            await self.metadata_extractor.connect_to_database(database_name)
            await self.performance_analyzer.connect_to_database(database_name)
            
            self.connection_info["current_database"] = database_name
            
            return f"✅ Connected to database: {database_name}\n\n" \
                   f"💡 You can now:\n" \
                   f"- Get model metadata with 'get_model_metadata'\n" \
                   f"- Optimize measures with 'optimize_dax_measure'\n" \
                   f"- Analyze performance with 'analyze_dax_performance'"
            
        except Exception as e:
            logger.error(f"Connect to database failed: {e}")
            return f"❌ Failed to connect to database: {str(e)}"
    
    async def _handle_get_metadata(self, arguments: Dict[str, Any]) -> str:
        """Get model metadata"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        try:
            # Check if a DAX query is provided to get focused metadata
            dax_query = arguments.get("dax_query")
            
            if dax_query:
                # Get focused metadata based on DAX query dependencies
                metadata = await self.metadata_extractor.get_dax_focused_metadata(dax_query)
                result = f"🎯 Focused Model Metadata for '{self.connection_info.get('current_database', 'Current Database')}'\n"
                result += "=" * 70 + "\n\n"
                result += f"� Based on DAX query dependencies\n\n"
            else:
                # Get full metadata
                metadata = await self.metadata_extractor.get_full_metadata()
                result = f"�📊 Full Model Metadata for '{self.connection_info.get('current_database', 'Current Database')}'\n"
                result += "=" * 60 + "\n\n"
            
            # Tables summary
            tables = metadata.get('tables', [])
            result += f"📋 Tables ({len(tables)}):\n"
            for table in tables[:10]:  # Show first 10
                result += f"  • {table['name']} ({table.get('row_count', 'Unknown')} rows)\n"
            if len(tables) > 10:
                result += f"  ... and {len(tables) - 10} more tables\n"
            result += "\n"
            
            # Columns summary (show more detail for focused metadata)
            columns = metadata.get('columns', [])
            result += f"📊 Columns ({len(columns)}):\n"
            if dax_query and len(columns) <= 20:
                # Show all columns for focused metadata if not too many
                for col in columns:
                    result += f"  • {col['table_name']}[{col['column_name']}] ({col.get('data_type', 'Unknown')})\n"
            else:
                # Show summary for full metadata
                table_col_counts = {}
                for col in columns:
                    table_name = col['table_name']
                    table_col_counts[table_name] = table_col_counts.get(table_name, 0) + 1
                
                for table_name, col_count in list(table_col_counts.items())[:5]:
                    result += f"  • {table_name}: {col_count} columns\n"
                if len(table_col_counts) > 5:
                    result += f"  ... and {len(table_col_counts) - 5} more tables\n"
            result += "\n"
            
            # Relationships summary
            relationships = metadata.get('relationships', [])
            result += f"🔗 Relationships ({len(relationships)}):\n"
            for rel in relationships[:5]:  # Show first 5
                result += f"  • {rel['from_table']}[{rel['from_column']}] → {rel['to_table']}[{rel['to_column']}]\n"
            if len(relationships) > 5:
                result += f"  ... and {len(relationships) - 5} more relationships\n"
            result += "\n"
            
            # Show query dependencies if focused metadata was used
            if dax_query and metadata.get('query_dependencies'):
                deps = metadata['query_dependencies']
                result += f"🎯 Query Dependencies ({len(deps)}):\n"
                for dep in deps[:10]:
                    result += f"  • {dep['Table Name']}[{dep['Column Name']}]\n"
                if len(deps) > 10:
                    result += f"  ... and {len(deps) - 10} more dependencies\n"
                result += "\n"
            
            if dax_query:
                result += "💡 This focused metadata includes only tables/columns relevant to your DAX query"
            else:
                result += "💡 Use 'get_model_metadata' with a dax_query parameter to get focused metadata"
            
            return result            
        except Exception as e:
            logger.error(f"Get metadata failed: {e}")
            return f"❌ Failed to get metadata: {str(e)}"
    
    async def _handle_optimize_measure(self, arguments: Dict[str, Any]) -> str:
        """Handle DAX measure optimization"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        measure_name = arguments.get("measure_name")
        dax_expression = arguments.get("dax_expression")
        max_iterations = arguments.get("max_iterations", 3)
        
        try:
            # First, define all measures in the DAX expression to make it self-contained
            logger.info(f"Defining measures for optimization of {measure_name}")
            
            # Create a simple DAX query to wrap the measure for dependency analysis
            test_query = f"EVALUATE ROW(\"Result\", {dax_expression})"
            defined_query = self.dax_analyzer.define_query_measures(test_query)
            
            # Extract the defined expression from the query
            if "DEFINE" in defined_query:
                # Extract just the measure definitions part
                defined_dax = defined_query
            else:
                defined_dax = dax_expression
            
            # Include uploaded file context
            context = self._get_uploaded_context()
            
            # Get focused metadata for the measure
            metadata = await self.metadata_extractor.get_dax_focused_metadata(test_query)
            
            optimization_result = await self.dax_analyzer.optimize_measure(
                measure_name, defined_dax, max_iterations, context
            )
            
            # Add metadata info to the result
            if metadata and metadata.get('query_dependencies'):
                deps_count = len(metadata['query_dependencies'])
                optimization_result['metadata_info'] = f"Analyzed {deps_count} query dependencies"
            
            return self._format_optimization_result(optimization_result)
            
        except Exception as e:
            logger.error(f"Optimize measure failed: {e}")
            return f"❌ Optimization failed: {str(e)}"
    
    async def _handle_analyze_performance(self, arguments: Dict[str, Any]) -> str:
        """Handle DAX performance analysis"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        dax_query = arguments.get("dax_query")
        
        try:
            analysis_result = await self.performance_analyzer.analyze_query(dax_query)
            
            return self._format_performance_analysis(analysis_result)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return f"❌ Performance analysis failed: {str(e)}"
    
    async def _handle_search_knowledge(self, arguments: Dict[str, Any]) -> str:
        """Handle knowledge base search"""
        query = arguments.get("query")
        limit = arguments.get("limit", 5)
        
        try:
            results = await self.kb.search(query, limit)
            
            if not results:
                return f"❌ No results found for '{query}'\n\n" \
                       f"💡 Try updating the knowledge base with 'update_knowledge_base'"
            
            response = f"🔍 DAX Knowledge Search Results for '{query}'\n"
            response += "=" * 50 + "\n\n"
            
            for i, result in enumerate(results, 1):
                response += f"{i}. 📖 {result['title']}\n"
                response += f"   📄 {result['snippet']}\n"
                response += f"   🔗 {result['url']}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"❌ Knowledge search failed: {str(e)}"
    
    async def _handle_update_knowledge_base(self, arguments: Dict[str, Any]) -> str:
        """Handle knowledge base update"""
        try:
            result = await self.kb.update_knowledge_base()
            return f"📚 Knowledge Base Update Complete\n{result}"
        except Exception as e:
            logger.error(f"KB update failed: {e}")
            return f"❌ Knowledge base update failed: {str(e)}"
    
    async def _handle_upload_file(self, arguments: Dict[str, Any]) -> str:
        """Handle file upload"""
        filename = arguments.get("filename")
        content = arguments.get("content")
        
        if not filename or not content:
            return "❌ Please provide both filename and content"
        
        self.uploaded_files[filename] = {
            "content": content,
            "upload_time": datetime.now().isoformat()
        }
        
        return f"✅ File '{filename}' uploaded successfully\n" \
               f"📄 Content length: {len(content)} characters\n" \
               f"💡 This content will be used as context for DAX optimization"
    
    async def _handle_compare_variants(self, arguments: Dict[str, Any]) -> str:
        """Handle DAX variant comparison"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        measure_name = arguments.get("measure_name")
        variants = arguments.get("variants", [])
        
        if len(variants) < 2:
            return "❌ Please provide at least 2 variants to compare"
        
        try:
            comparison_result = await self.performance_analyzer.compare_variants(
                measure_name, variants
            )
            
            return self._format_comparison_result(comparison_result)
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {e}")
            return f"❌ Variant comparison failed: {str(e)}"
    
    async def _handle_get_server_info(self, arguments: Dict[str, Any]) -> str:
        """Handle server info request"""
        if not self.is_connected:
            return "❌ Not connected. Please login first."
        
        try:
            server_info = await self.dax_analyzer.get_detailed_server_info()
            
            result = "🖥️ Analysis Services Server Information\n"
            result += "=" * 40 + "\n\n"
            
            for key, value in server_info.items():
                result += f"📋 {key}: {value}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Get server info failed: {e}")
            return f"❌ Failed to get server info: {str(e)}"
    
    def _get_uploaded_context(self) -> str:
        """Get context from uploaded files"""
        if not self.uploaded_files:
            return ""
        
        context = "\n📁 Uploaded File Context:\n"
        for filename, file_info in self.uploaded_files.items():
            context += f"\n--- {filename} ---\n"
            content = file_info['content']
            if len(content) > 3000:
                context += content[:3000] + "\n[... content truncated ...]\n"
            else:
                context += content + "\n"
        
        return context
    
    def _format_optimization_result(self, result: Dict[str, Any]) -> str:
        """Format optimization results for display"""
        output = f"🚀 DAX Optimization Results: '{result['measure_name']}'\n"
        output += "=" * 50 + "\n\n"
        
        if result.get('best_variant'):
            best = result['best_variant']
            output += "🎉 Best Optimization Found:\n"
            output += f"⚡ Performance Improvement: {best['improvement_percent']:.1f}%\n"
            output += f"⏱️ Execution Time: {best['duration_ms']:.2f}ms\n"
            output += f"✅ Results Match: {best['results_match']}\n\n"
            output += f"📝 Optimized DAX:\n```dax\n{best['dax']}\n```\n\n"
        else:
            output += "❌ No improvements found\n\n"
        
        # Show iteration details
        if result.get('iterations'):
            output += "📊 Optimization Attempts:\n"
            for i, iteration in enumerate(result['iterations'], 1):
                output += f"  {i}. {iteration.get('improvement_percent', 0):.1f}% improvement "
                output += f"({iteration.get('duration_ms', 0):.2f}ms)\n"
        
        return output
    
    def _format_performance_analysis(self, result: Dict[str, Any]) -> str:
        """Format performance analysis results"""
        output = "🔍 DAX Performance Analysis\n"
        output += "=" * 30 + "\n\n"
        
        output += f"⏱️ Execution Time: {result.get('duration_ms', 0):.2f}ms\n"
        output += f"📊 Rows Returned: {result.get('row_count', 0)}\n"
        output += f"💾 Memory Usage: {result.get('memory_mb', 0):.2f}MB\n\n"
        
        # Performance rating
        duration = result.get('duration_ms', 0)
        if duration < 100:
            output += "⚡ Performance: Excellent (< 100ms)\n"
        elif duration < 500:
            output += "✅ Performance: Good (100-500ms)\n"
        elif duration < 2000:
            output += "⚠️ Performance: Moderate (500ms-2s)\n"
        else:
            output += "🐌 Performance: Slow (> 2s) - Consider optimization\n"
        
        return output
    
    def _format_comparison_result(self, result: Dict[str, Any]) -> str:
        """Format variant comparison results"""
        output = f"⚖️ DAX Variant Comparison: '{result['measure_name']}'\n"
        output += "=" * 40 + "\n\n"
        
        variants = result.get('variants', [])
        if variants:
            # Find fastest
            fastest = min(variants, key=lambda x: x.get('duration_ms', float('inf')))
            output += f"🏆 Fastest: {fastest['name']} ({fastest.get('duration_ms', 0):.2f}ms)\n\n"
            
            # Show all variants
            for variant in sorted(variants, key=lambda x: x.get('duration_ms', 0)):
                output += f"📊 {variant['name']}:\n"
                output += f"   ⏱️ {variant.get('duration_ms', 0):.2f}ms\n"
                output += f"   ✅ Results Match: {variant.get('results_match', False)}\n\n"
        
        return output
    
    async def run(self):
        """Run the MCP server"""
        try:
            logger.info("🚀 Starting DAX Optimizer MCP Server...")
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="dax-optimizer-mcp-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("DAX Optimizer server shutting down")


async def main():
    """Main entry point"""
    server = DAXOptimizerMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("DAX Optimizer server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)