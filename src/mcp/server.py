import asyncio
from typing import Any, Dict, List, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import os
from dotenv import load_dotenv
import json
from datetime import datetime, date
from decimal import Decimal
import sys
import clr
import openai
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Configure logging to stderr for MCP debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Updated imports
try:
    from mcp.types import Tool, TextContent, Resource, Prompt
    from mcp.server.types import ToolResult
except ImportError:
    from mcp.types import Tool, TextContent

# Custom JSON encoder for Power BI data types
class PowerBIJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Power BI data types"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)

def safe_json_dumps(data, indent=2):
    """Safely serialize data containing datetime and other non-JSON types"""
    return json.dumps(data, indent=indent, cls=PowerBIJSONEncoder)

def clean_dax_query(dax_query: str) -> str:
    """Remove HTML/XML tags and other artifacts from DAX queries"""
    # Remove HTML/XML tags like <oii>, </oii>, etc.
    cleaned = re.sub(r'<[^>]+>', '', dax_query)
    # Remove any remaining angle brackets
    cleaned = cleaned.replace('<', '').replace('>', '')
    # Clean up extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned
    
# Load environment variables
load_dotenv()

# Add the path to the ADOMD.NET library
adomd_paths = [
    r"C:\Program Files\Microsoft.NET\ADOMD.NET\160",
    r"C:\Program Files\Microsoft.NET\ADOMD.NET\150",
    r"C:\Program Files (x86)\Microsoft.NET\ADOMD.NET\160",
    r"C:\Program Files (x86)\Microsoft.NET\ADOMD.NET\150"
]

adomd_loaded = False
for path in adomd_paths:
    if os.path.exists(path):
        try:
            sys.path.append(path)
            clr.AddReference("Microsoft.AnalysisServices.AdomdClient")
            adomd_loaded = True
            logger.info(f"Loaded ADOMD.NET from {path}")
            break
        except Exception as e:
            logger.debug(f"Failed to load ADOMD.NET from {path}: {e}")
            continue

if not adomd_loaded:
    logger.error("Could not load ADOMD.NET library")
    raise ImportError("Could not load ADOMD.NET library. Please install SSMS or ADOMD.NET client.")

from pyadomd import Pyadomd
from Microsoft.AnalysisServices.AdomdClient import AdomdSchemaGuid


class PowerBIConnector:
    def __init__(self):
        self.connection_string = None
        self.connected = False
        self.tables = []
        self.metadata = {}
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def connect(self, xmla_endpoint: str, tenant_id: str, client_id: str, 
                client_secret: str, initial_catalog: str) -> bool:
        """Establish connection to Power BI dataset"""
        self.connection_string = (
            f"Provider=MSOLAP;"
            f"Data Source={xmla_endpoint};"
            f"Initial Catalog={initial_catalog};"
            f"User ID=app:{client_id}@{tenant_id};"
            f"Password={client_secret};"
        )
        
        try:
            # Test connection
            with Pyadomd(self.connection_string) as conn:
                self.connected = True
                logger.info(f"Connected to Power BI dataset: {initial_catalog}")
                # Don't discover tables during connection to speed up
                return True
        except Exception as e:
            self.connected = False
            logger.error(f"Connection failed: {str(e)}")
            raise Exception(f"Connection failed: {str(e)}")
    
    def discover_tables(self) -> List[str]:
        """Discover all user-facing tables in the dataset"""
        if not self.connected:
            raise Exception("Not connected to Power BI")
            
        # Return cached tables if already discovered
        if self.tables:
            return self.tables
            
        tables_list = []
        try:
            with Pyadomd(self.connection_string) as pyadomd_conn:
                adomd_connection = pyadomd_conn.conn
                tables_dataset = adomd_connection.GetSchemaDataSet(AdomdSchemaGuid.Tables, None)
                
                if tables_dataset and tables_dataset.Tables.Count > 0:
                    schema_table = tables_dataset.Tables[0]
                    for row in schema_table.Rows:
                        table_name = row["TABLE_NAME"]
                        if (not table_name.startswith("$") and 
                            not table_name.startswith("DateTableTemplate_") and 
                            not row["TABLE_SCHEMA"] == "$SYSTEM"):
                            tables_list.append(table_name)
                            
            self.tables = tables_list
            logger.info(f"Discovered {len(tables_list)} tables")
            return tables_list
        except Exception as e:
            logger.error(f"Failed to discover tables: {str(e)}")
            raise Exception(f"Failed to discover tables: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        if not self.connected:
            raise Exception("Not connected to Power BI")
            
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Try to get column information
                try:
                    dax_query = f"EVALUATE TOPN(1, '{table_name}')"
                    cursor.execute(dax_query)
                    columns = [desc[0] for desc in cursor.description]
                    cursor.close()
                    
                    return {
                        "table_name": table_name,
                        "type": "data_table",
                        "columns": columns
                    }
                except:
                    # This might be a measure table
                    cursor.close()
                    return self.get_measures_for_table(table_name)
                    
        except Exception as e:
            logger.error(f"Failed to get schema for table '{table_name}': {str(e)}")
            raise Exception(f"Failed to get schema for table '{table_name}': {str(e)}")
    
    def get_measures_for_table(self, table_name: str) -> Dict[str, Any]:
        """Get measures for a measure table"""
        try:
            with Pyadomd(self.connection_string) as conn:
                # Get table ID
                id_cursor = conn.cursor()
                id_query = f"SELECT [ID] FROM $SYSTEM.TMSCHEMA_TABLES WHERE [Name] = '{table_name}'"
                id_cursor.execute(id_query)
                table_id_result = id_cursor.fetchone()
                id_cursor.close()
                
                if not table_id_result:
                    return {"table_name": table_name, "type": "unknown", "measures": []}
                
                table_id = table_id_result[0]
                
                # Get measures
                measure_cursor = conn.cursor()
                measure_query = f"SELECT [Name], [Expression] FROM $SYSTEM.TMSCHEMA_MEASURES WHERE [TableID] = {table_id} ORDER BY [Name]"
                measure_cursor.execute(measure_query)
                measures = measure_cursor.fetchall()
                measure_cursor.close()
                
                return {
                    "table_name": table_name,
                    "type": "measure_table",
                    "measures": [{"name": m[0], "dax": m[1]} for m in measures]
                }
                
        except Exception as e:
            logger.error(f"Failed to get measures for table '{table_name}': {str(e)}")
            return {"table_name": table_name, "type": "error", "error": str(e)}
    
    def execute_dax_query(self, dax_query: str) -> List[Dict[str, Any]]:
        """Execute a DAX query and return results"""
        if not self.connected:
            raise Exception("Not connected to Power BI")
            
        # Clean the DAX query
        cleaned_query = clean_dax_query(dax_query)
        logger.info(f"Executing DAX query: {cleaned_query}")
            
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(cleaned_query)
                
                headers = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(zip(headers, row)))
                
                logger.info(f"Query returned {len(results)} rows")
                return results
                
        except Exception as e:
            logger.error(f"DAX query failed: {str(e)}")
            raise Exception(f"DAX query failed: {str(e)}")
    
    def get_sample_data(self, table_name: str, num_rows: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        dax_query = f"EVALUATE TOPN({num_rows}, '{table_name}')"
        return self.execute_dax_query(dax_query)


class DataAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.context = {
            "tables": [],
            "schemas": {},
            "sample_data": {}
        }
    
    def set_data_context(self, tables: List[str], schemas: Dict[str, Any], 
                        sample_data: Dict[str, List[Dict[str, Any]]]):
        """Set the data context for the analyzer"""
        self.context = {
            "tables": tables,
            "schemas": schemas,
            "sample_data": sample_data
        }
    
    def generate_dax_query(self, user_question: str) -> str:
        """Generate a DAX query based on user question"""
        prompt = f"""
        You are a Power BI DAX expert. Generate a DAX query to answer the following question.
        
        Available tables and their schemas:
        {safe_json_dumps(self.context['schemas'], indent=2)}
        
        Sample data for reference:
        {safe_json_dumps(self.context['sample_data'], indent=2)}
        
        User question: {user_question}
        
        IMPORTANT RULES:
        1. Generate only the DAX query without any explanation
        2. Do NOT use any HTML or XML tags in the query
        3. Do NOT use angle brackets < or > except for DAX operators
        4. Use only valid DAX syntax
        5. Reference only columns that exist in the schema
        6. The query should be executable as-is
        
        Example format:
        EVALUATE SUMMARIZE(Sales, Product[Category], "Total", SUM(Sales[Amount]))
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a DAX query expert. Generate only valid, clean DAX queries without any markup or formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        query = response.choices[0].message.content.strip()
        # Clean any remaining artifacts
        return clean_dax_query(query)
    
    def interpret_results(self, user_question: str, query_results: List[Dict[str, Any]], 
                         dax_query: str) -> str:
        """Interpret query results and provide a natural language answer"""
        prompt = f"""
        You are a data analyst helping users understand their Power BI data.
        
        User question: {user_question}
        
        DAX query executed: {dax_query}
        
        Query results:
        {safe_json_dumps(query_results, indent=2)}
        
        Provide a clear, concise answer to the user's question based on the results.
        Include relevant numbers and insights. Format the response in a user-friendly way.
        Do not use any HTML or XML markup in your response.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst providing insights from Power BI data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def suggest_questions(self) -> List[str]:
        """Suggest relevant questions based on available data"""
        prompt = f"""
        Based on the following Power BI dataset structure, suggest 5 interesting questions a user might ask:
        
        Tables and schemas:
        {safe_json_dumps(self.context['schemas'], indent=2)}
        
        Generate 5 diverse questions that would showcase different aspects of the data.
        Return only the questions as a JSON array.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst suggesting interesting questions about data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        try:
            questions = json.loads(response.choices[0].message.content)
            return questions
        except:
            return ["What are the total sales?", "Show me the top 10 products", 
                   "What is the trend over time?", "Which region has the highest revenue?",
                   "What are the key metrics?"]


class PowerBIMCPServer:
    def __init__(self):
        self.server = Server("powerbi-mcp-server")
        self.connector = PowerBIConnector()
        self.analyzer = None
        self.is_connected = False
        self.connection_lock = threading.Lock()
        
        # Setup server handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="connect_powerbi",
                    description="Connect to a Power BI dataset using XMLA endpoint",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "xmla_endpoint": {"type": "string", "description": "Power BI XMLA endpoint URL"},
                            "tenant_id": {"type": "string", "description": "Azure AD Tenant ID"},
                            "client_id": {"type": "string", "description": "Service Principal Client ID"},
                            "client_secret": {"type": "string", "description": "Service Principal Client Secret"},
                            "initial_catalog": {"type": "string", "description": "Dataset name"}
                        },
                        "required": ["xmla_endpoint", "tenant_id", "client_id", "client_secret", "initial_catalog"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all available tables in the connected Power BI dataset",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_table_info",
                    description="Get detailed information about a specific table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Name of the table"}
                        },
                        "required": ["table_name"]
                    }
                ),
                Tool(
                    name="query_data",
                    description="Ask a question about the data in natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Your question about the data"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="execute_dax",
                    description="Execute a custom DAX query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dax_query": {"type": "string", "description": "DAX query to execute"}
                        },
                        "required": ["dax_query"]
                    }
                ),
                Tool(
                    name="suggest_questions",
                    description="Get suggestions for interesting questions to ask about the data",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """Return empty list of resources - stub implementation"""
            return []
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """Return empty list of prompts - stub implementation"""
            return []
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
            """Handle tool calls and return results as TextContent"""
            try:
                logger.info(f"Handling tool call: {name}")
                
                if name == "connect_powerbi":
                    result = await self._handle_connect(arguments)
                elif name == "list_tables":
                    result = await self._handle_list_tables()
                elif name == "get_table_info":
                    result = await self._handle_get_table_info(arguments)
                elif name == "query_data":
                    result = await self._handle_query_data(arguments)
                elif name == "execute_dax":
                    result = await self._handle_execute_dax(arguments)
                elif name == "suggest_questions":
                    result = await self._handle_suggest_questions()
                else:
                    logger.warning(f"Unknown tool: {name}")
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
                # Convert string result to TextContent
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Error executing {name}: {str(e)}", exc_info=True)
                return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
    
    async def _handle_connect(self, arguments: Dict[str, Any]) -> str:
        """Handle connection to Power BI"""
        try:
            with self.connection_lock:
                # Connect to Power BI
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.connector.connect,
                    arguments["xmla_endpoint"],
                    arguments["tenant_id"],
                    arguments["client_id"],
                    arguments["client_secret"],
                    arguments["initial_catalog"]
                )
                
                # Initialize the analyzer with OpenAI API key
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return "OpenAI API key not found in environment variables"
                
                self.analyzer = DataAnalyzer(api_key)
                self.is_connected = True
                
                # Discover tables in background
                asyncio.create_task(self._async_prepare_context())
                
                return f"Successfully connected to Power BI dataset '{arguments['initial_catalog']}'. Discovering tables..."
                
        except Exception as e:
            self.is_connected = False
            logger.error(f"Connection failed: {str(e)}")
            return f"Connection failed: {str(e)}"
    
    async def _async_prepare_context(self):
        """Prepare data context asynchronously"""
        try:
            # Discover tables
            tables = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.discover_tables
            )
            
            schemas = {}
            sample_data = {}
            
            # Get schemas for first 5 tables only to speed up
            for table in tables[:5]:
                try:
                    schema = await asyncio.get_event_loop().run_in_executor(
                        None, self.connector.get_table_schema, table
                    )
                    schemas[table] = schema
                    
                    if schema["type"] == "data_table":
                        samples = await asyncio.get_event_loop().run_in_executor(
                            None, self.connector.get_sample_data, table, 3
                        )
                        sample_data[table] = samples
                except Exception as e:
                    logger.warning(f"Failed to get schema for table {table}: {e}")
            
            self.analyzer.set_data_context(tables, schemas, sample_data)
            logger.info(f"Context prepared with {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
    
    async def _handle_list_tables(self) -> str:
        """List all available tables"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first using 'connect_powerbi'."
        
        try:
            tables = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.discover_tables
            )
            
            if not tables:
                return "No tables found in the dataset."
            
            table_list = "\n".join([f"- {table}" for table in tables])
            return f"Available tables:\n{table_list}"
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return f"Error listing tables: {str(e)}"
    
    async def _handle_get_table_info(self, arguments: Dict[str, Any]) -> str:
        """Get information about a specific table"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        table_name = arguments.get("table_name")
        if not table_name:
            return "Please provide a table name."
        
        try:
            schema = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.get_table_schema, table_name
            )
            
            if schema["type"] == "data_table":
                sample_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.connector.get_sample_data, table_name, 5
                )
                result = f"Table: {table_name}\nType: Data Table\nColumns: {', '.join(schema['columns'])}\n\nSample data:\n"
                result += safe_json_dumps(sample_data, indent=2)
            elif schema["type"] == "measure_table":
                result = f"Table: {table_name}\nType: Measure Table\nMeasures:\n"
                for measure in schema["measures"]:
                    result += f"\n- {measure['name']}:\n  DAX: {measure['dax']}\n"
            else:
                result = f"Table: {table_name}\nType: {schema['type']}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return f"Error getting table info: {str(e)}"
    
    async def _handle_query_data(self, arguments: Dict[str, Any]) -> str:
        """Handle natural language queries about data"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        if not self.analyzer:
            return "Data analyzer not initialized."
        
        question = arguments.get("question")
        if not question:
            return "Please provide a question."
        
        try:
            # Generate DAX query
            dax_query = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.generate_dax_query, question
            )
            
            # Execute the query
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax_query
            )
            
            # Interpret results
            interpretation = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.interpret_results, question, results, dax_query
            )
            
            response = f"Question: {question}\n\nDAX Query:\n{dax_query}\n\nAnswer:\n{interpretation}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    async def _handle_execute_dax(self, arguments: Dict[str, Any]) -> str:
        """Execute a custom DAX query"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        dax_query = arguments.get("dax_query")
        if not dax_query:
            return "Please provide a DAX query."
        
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax_query
            )
            return safe_json_dumps(results, indent=2)
        except Exception as e:
            logger.error(f"DAX execution error: {e}")
            return f"DAX execution error: {str(e)}"
    
    async def _handle_suggest_questions(self) -> str:
        """Suggest interesting questions about the data"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        if not self.analyzer:
            return "Data analyzer not initialized. Please wait for tables to be discovered."
        
        try:
            questions = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.suggest_questions
            )
            result = "Here are some questions you might want to ask:\n\n"
            for i, q in enumerate(questions, 1):
                result += f"{i}. {q}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return f"Error generating suggestions: {str(e)}"
    
    async def run(self):
        """Run the MCP server"""
        try:
            logger.info("Starting Power BI MCP Server...")
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="powerbi-mcp-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except anyio.BrokenResourceError:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Server shutting down")


# Main entry point
async def main():
    server = PowerBIMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
